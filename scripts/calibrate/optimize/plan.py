#!/usr/bin/env python3
"""Choose how many chunks to split a workload into, and what to request per job.

Given a workload (N sequences with their lengths) and the fitted cost model,
this searches chunking strategies and reports the cost/time trade-off, then
recommends per-job SLURM resources.

Why this is an optimization and not a preference
------------------------------------------------
Every job pays `overhead` — container start, model weights off Lustre, CUDA
init — no matter how little work it does. So:

    total GPU time  = n_chunks · overhead + (work, fixed by the workload)
    makespan        ≈ (per-chunk time) · ceil(n_chunks / max_concurrent)

Splitting finer buys wall-clock and costs GPU-hours; splitting coarser does the
reverse. There is no single right answer, so the output is a frontier plus two
labelled picks: cheapest, and fastest-that-is-still-reasonably-efficient.

Three things the search accounts for that a fixed `max_files_per_job` cannot:

  * Length heterogeneity. A chunk's VRAM is set by its LONGEST sequence, so one
    2000-residue protein forces a whole 50-sequence chunk onto a big GPU.
  * The balance/uniformity trade-off. Sorting by length makes each chunk's
    resources tight and predictable but leaves the long-sequence chunk running
    long after the others finished; dealing sequences round-robin evens out the
    runtimes but puts a long sequence in every chunk, so every chunk needs the
    big-GPU headroom. Both are evaluated.
  * Feasibility. Plans whose predicted VRAM or host RAM exceed the target GPU
    are rejected rather than silently recommended.

Predictions inherit the model's uncertainty; where the model says cost-vs-length
is unconstrained, treat chunk sizing as provisional.

Usage:
    python -m scripts.calibrate.optimize.plan --model cost_model.yaml \
        --stage esmfold --fasta-dir /path/to/fastas [--max-concurrent 10]
    # or without real inputs:
    python -m scripts.calibrate.optimize.plan --model cost_model.yaml \
        --stage esmc --n-seqs 110 --mean-len 284
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import yaml

# Relative price of a GPU-second per card. Defaults are ordinal (an H100 costs
# more than a MIG slice); override with --gpu-costs to match real billing
# weights. Only ratios matter for ranking plans.
GPU_COST = {"a100_mig": 0.3, "a100": 1.0, "a100_80": 1.4, "h100": 2.5}
GPU_VRAM_GB = {"a100_mig": 20, "a100": 40, "a100_80": 80, "h100": 80}
GPU_PARTITION = {"a100_mig": "kempner_requeue", "a100": "kempner",
                 "a100_80": "kempner", "h100": "kempner_h100"}
# Below this SM utilization a stage is not compute-bound, so a slower card costs
# little; at or above it, downgrading just makes the job run longer.
COMPUTE_BOUND_PCT = 50.0

MEM_SAFETY = 1.3
TIME_SAFETY = 1.5
VRAM_SAFETY = 1.2
MIN_RUNTIME_MIN = 15


# --- workload --------------------------------------------------------------

def lengths_from_fasta_dir(d: Path) -> list[int]:
    out = []
    for p in sorted(glob.glob(os.path.join(str(d), "*.fasta"))):
        seq = "".join(l.strip() for l in open(p) if not l.startswith(">"))
        if seq:
            out.append(len(seq))
    return out


def synthetic_lengths(n: int, mean_len: int, max_len: int | None) -> list[int]:
    """Flat workload when no real inputs are given; a spread if max_len is set."""
    if not max_len or max_len <= mean_len:
        return [mean_len] * n
    lo = max(1, 2 * mean_len - max_len)
    return [round(lo + (max_len - lo) * i / max(1, n - 1)) for i in range(n)]


# --- chunking strategies ---------------------------------------------------

def chunks_sequential(lengths: list[int], k: int) -> list[list[int]]:
    """Input order, fixed chunk size — what max_files_per_job does today."""
    return [lengths[i:i + k] for i in range(0, len(lengths), k)]


def chunks_length_binned(lengths: list[int], n_chunks: int) -> list[list[int]]:
    """Sort by length, split into contiguous equal-count blocks.

    Each chunk is length-homogeneous, so its VRAM and runtime are predictable
    and can be sized tightly — at the price of one slow chunk of long sequences.
    """
    s = sorted(lengths)
    n = len(s)
    out, start = [], 0
    for i in range(n_chunks):
        end = round(n * (i + 1) / n_chunks)
        if end > start:
            out.append(s[start:end])
        start = end
    return out


def chunks_balanced(lengths: list[int], n_chunks: int, cost_of) -> list[list[int]]:
    """Greedy longest-processing-time: even out predicted work per chunk.

    Gives near-equal runtimes (good makespan) but scatters long sequences across
    every chunk, so each one needs the big-GPU headroom.
    """
    bins: list[list[int]] = [[] for _ in range(n_chunks)]
    loads = [0.0] * n_chunks
    for L in sorted(lengths, reverse=True):
        # Tie-break on bin occupancy. Without it a stage whose per-sequence cost
        # fits to ~0 (MSA — one database scan serves the whole chunk) leaves
        # every load at 0.0, so min() returns bin 0 every time and the entire
        # workload collapses into a single chunk no matter what n_chunks says.
        i = min(range(n_chunks), key=lambda j: (loads[j], len(bins[j])))
        bins[i].append(L)
        loads[i] += cost_of(L)
    return [b for b in bins if b]


# --- evaluation ------------------------------------------------------------

class Model:
    def __init__(self, block: dict):
        t = block.get("time", {}) or {}
        self.overhead = float(t.get("overhead_s", 0.0))
        self.scale = float(t.get("scale", 1.0))
        self.a = float(t.get("a", 0.0))
        self.b = float(t.get("b", 0.0))
        self.c = float(t.get("c", 0.0))
        v = block.get("vram_gb", {}) or {}
        self.v0 = float(v.get("base", 0.0))
        self.v1 = float(v.get("per_len", 0.0))
        self.v2 = float(v.get("per_len2", 0.0))
        h = block.get("host_gb", {}) or {}
        self.h0 = float(h.get("base", 0.0))
        self.h1 = float(h.get("per_seq", 0.0))
        self.h2 = float(h.get("per_residue", 0.0))
        gu = block.get("gpu_util_pct", {}) or {}
        self.gpu_util = gu.get("median")
        self.notes = block.get("notes", [])
        self.fit_range = block.get("fitted_len_range")
        self.fit_n_range = block.get("fitted_n_range")

    def seq_cost(self, L: float) -> float:
        return self.scale * (self.a + self.b * L + self.c * L * L)

    def job_time(self, chunk: list[int]) -> float:
        return self.overhead + sum(self.seq_cost(L) for L in chunk)

    def job_vram(self, chunk: list[int]) -> float:
        m = max(chunk) if chunk else 0
        return self.v0 + self.v1 * m + self.v2 * m * m

    def job_host(self, chunk: list[int]) -> float:
        return self.h0 + self.h1 * len(chunk) + self.h2 * sum(chunk)


def makespan(times: list[float], concurrency: int) -> float:
    """List-scheduling makespan: each job starts when a slot frees up."""
    if not times:
        return 0.0
    slots = [0.0] * max(1, concurrency)
    for t in sorted(times, reverse=True):
        i = min(range(len(slots)), key=lambda j: slots[j])
        slots[i] += t
    return max(slots)


def pick_gpu(vram_need: float, gpu_util: float | None, allowed: list[str]) -> str | None:
    """Smallest card that fits — unless the stage is compute-bound, in which case
    a smaller card would just run longer for the same work."""
    fits = [g for g in allowed if GPU_VRAM_GB.get(g, 0) >= vram_need]
    if not fits:
        return None
    if gpu_util is not None and gpu_util >= COMPUTE_BOUND_PCT:
        return max(fits, key=lambda g: GPU_COST.get(g, 1.0))
    return min(fits, key=lambda g: GPU_COST.get(g, 1.0))


def evaluate(chunks: list[list[int]], m: Model, concurrency: int,
             allowed_gpus: list[str], host_cap_gb: float,
             max_runtime_min: float) -> dict | None:
    times = [m.job_time(c) for c in chunks]
    vram = [m.job_vram(c) for c in chunks]
    host = [m.job_host(c) for c in chunks]

    vram_need = max(vram) * VRAM_SAFETY if vram else 0.0
    gpu = pick_gpu(vram_need, m.gpu_util, allowed_gpus)
    if gpu is None:
        return None  # no card can hold the biggest chunk
    if max(host) * MEM_SAFETY > host_cap_gb:
        return None
    # A job that cannot finish inside the partition's wall-clock limit is not a
    # plan, however cheap it looks on paper.
    if max(times) * TIME_SAFETY / 60.0 > max_runtime_min:
        return None

    total_gpu_s = sum(times)
    work_s = total_gpu_s - m.overhead * len(chunks)
    price = GPU_COST.get(gpu, 1.0)
    return {
        "n_chunks": len(chunks),
        "chunk_sizes": (min(len(c) for c in chunks), max(len(c) for c in chunks)),
        "gpu": gpu,
        "cost": total_gpu_s * price / 3600.0,        # weighted GPU-hours
        "gpu_hours": total_gpu_s / 3600.0,
        "makespan_min": makespan(times, concurrency) / 60.0,
        "overhead_frac": (m.overhead * len(chunks) / total_gpu_s) if total_gpu_s else 0.0,
        "efficiency": (work_s / total_gpu_s) if total_gpu_s else 0.0,
        "mem_mb": int(math.ceil(max(host) * MEM_SAFETY * 1024 / 1000) * 1000) or 4000,
        "runtime_min": max(MIN_RUNTIME_MIN,
                           int(math.ceil(max(times) * TIME_SAFETY / 60))),
        "vram_peak_gb": max(vram),
        "host_peak_gb": max(host),
    }


def search(lengths: list[int], m: Model, concurrency: int, allowed_gpus: list[str],
           host_cap_gb: float, max_chunks: int, max_runtime_min: float) -> list[dict]:
    n = len(lengths)
    out = []
    seen = set()
    for nc in range(1, min(n, max_chunks) + 1):
        k = math.ceil(n / nc)
        for name, chunks in (
            ("sequential", chunks_sequential(lengths, k)),
            ("length-binned", chunks_length_binned(lengths, nc)),
            ("balanced", chunks_balanced(lengths, nc, m.seq_cost)),
        ):
            if not chunks:
                continue
            key = (name, len(chunks), tuple(sorted(len(c) for c in chunks)),
                   tuple(sorted(max(c) for c in chunks)))
            if key in seen:
                continue
            seen.add(key)
            ev = evaluate(chunks, m, concurrency, allowed_gpus, host_cap_gb,
                          max_runtime_min)
            if ev:
                ev["strategy"] = name
                out.append(ev)
    return out


def best_value(plans: list[dict], tol: float = 0.05) -> dict:
    """Fastest plan whose cost is within `tol` of the cheapest.

    Picking the strict cost minimum is a bad default: when overhead is small
    relative to the work, cost is nearly flat across chunk counts, and the
    "cheapest" plan is then an arbitrarily serial one that runs for days to save
    a rounding error. Spending a few percent to finish several times sooner is
    almost always the intended trade.
    """
    cheapest = min(p["cost"] for p in plans)
    near = [p for p in plans if p["cost"] <= cheapest * (1 + tol)]
    return min(near, key=lambda p: p["makespan_min"])


def max_feasible_length(m: Model, vram_gb: float) -> int:
    """Longest single sequence that still fits on a card of `vram_gb`.

    A chunk's VRAM is set by its longest member, so any sequence past this point
    makes every chunk containing it unschedulable — it is not a chunking problem
    and no amount of re-chunking fixes it. Solved by bisection so the VRAM model
    can be any monotonic shape.
    """
    budget = vram_gb / VRAM_SAFETY
    if m.job_vram([1]) > budget:
        return 0
    lo, hi = 1, 100_000
    if m.job_vram([hi]) <= budget:
        return hi
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if m.job_vram([mid]) <= budget:
            lo = mid
        else:
            hi = mid
    return lo


def pareto(plans: list[dict]) -> list[dict]:
    """Plans not beaten on BOTH cost and makespan."""
    out = []
    for p in plans:
        if not any(q is not p and q["cost"] <= p["cost"]
                   and q["makespan_min"] <= p["makespan_min"]
                   and (q["cost"] < p["cost"] or q["makespan_min"] < p["makespan_min"])
                   for q in plans):
            out.append(p)
    return sorted(out, key=lambda p: p["cost"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--fasta-dir", type=Path)
    ap.add_argument("--n-seqs", type=int)
    ap.add_argument("--mean-len", type=int, default=300)
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--max-concurrent", type=int, default=8,
                    help="concurrent SLURM jobs this stage may hold (default 8: "
                         "the agreed cap on this shared allocation — planning "
                         "against more produces makespans you cannot achieve)")
    ap.add_argument("--max-chunks", type=int, default=64)
    ap.add_argument("--host-cap-gb", type=float, default=750.0,
                    help="largest host RAM a node can give a job")
    ap.add_argument("--max-runtime-min", type=float, default=720.0,
                    help="partition wall-clock limit; plans needing longer per "
                         "job are rejected (default 720 = 12 h)")
    ap.add_argument("--gpus", default="a100_mig,a100,a100_80,h100")
    ap.add_argument("--makespan-budget-min", type=float,
                    help="if set, recommend the cheapest plan finishing within this")
    ap.add_argument("--out", type=Path, help="write the recommended plan as YAML")
    args = ap.parse_args()

    model_all = yaml.safe_load(args.model.read_text())
    if args.stage not in model_all:
        print(f"stage {args.stage!r} not in model (have: "
              f"{', '.join(sorted(model_all))})", file=sys.stderr)
        return 2
    m = Model(model_all[args.stage])

    if args.fasta_dir:
        lengths = lengths_from_fasta_dir(args.fasta_dir)
        src = str(args.fasta_dir)
    elif args.n_seqs:
        lengths = synthetic_lengths(args.n_seqs, args.mean_len, args.max_len)
        src = f"synthetic n={args.n_seqs} mean_len={args.mean_len}"
    else:
        print("need --fasta-dir or --n-seqs", file=sys.stderr)
        return 2
    if not lengths:
        print("no sequences found", file=sys.stderr)
        return 1

    allowed = [g.strip() for g in args.gpus.split(",") if g.strip()]

    ls = sorted(lengths)
    print(f"stage: {args.stage}   workload: {len(lengths)} seqs from {src}")
    print(f"  length  min {ls[0]}  mean {sum(ls)/len(ls):.0f}  "
          f"p95 {ls[int(0.95*len(ls))-1]}  max {ls[-1]}")
    print(f"  model   overhead {m.overhead:.0f}s/job   "
          f"GPU util (observed median) "
          f"{m.gpu_util if m.gpu_util is not None else 'n/a'}%")
    for n in m.notes:
        print(f"  ! {n}")

    # Sequences too long for any allowed card are a filtering problem, not a
    # chunking problem: one of them poisons every chunk it lands in.
    biggest = max((GPU_VRAM_GB.get(g, 0) for g in allowed), default=0)
    cap = max_feasible_length(m, biggest)
    too_long = [L for L in ls if L > cap]
    if too_long:
        pct = 100.0 * len(too_long) / len(ls)
        print(f"\n  ** {len(too_long)} sequence(s) ({pct:.1f}%) exceed ~{cap} aa, "
              f"the longest that fits on the biggest allowed GPU "
              f"({biggest:.0f} GB).")
        print(f"     Longest are {too_long[-min(5,len(too_long)):]}. A chunk takes "
              f"its VRAM from its longest member, so each of these would stall or "
              f"OOM every chunk it joins.")
        print(f"     Planning for the remaining {len(ls)-len(too_long)}; run the "
              f"long ones separately or drop them.")
        lengths = [L for L in ls if L <= cap]
        if not lengths:
            print("\nnothing left to plan.", file=sys.stderr)
            return 1

    if m.fit_range:
        lo, hi = m.fit_range
        beyond = [L for L in lengths if L > hi]
        if beyond:
            print(f"\n  ! {len(beyond)} sequence(s) are longer than anything the "
                  f"model was fit on ({lo}-{hi} aa); their cost is extrapolated "
                  f"and may be wrong.")

    plans = search(lengths, m, args.max_concurrent, allowed,
                   args.host_cap_gb, args.max_chunks, args.max_runtime_min)
    if not plans:
        print("\nno feasible plan — every option exceeded VRAM, host RAM, or "
              f"the {args.max_runtime_min:.0f} min wall-clock limit",
              file=sys.stderr)
        return 1

    def fmt_span(v: float) -> str:
        return f"{v:.1f}m" if v < 10 else f"{v:.0f}m"

    front = pareto(plans)
    shown = front if len(front) <= 12 else (
        front[:6] + front[len(front) // 2 - 1:len(front) // 2 + 1] + front[-4:])
    print(f"\ncost / time frontier  (max {args.max_concurrent} concurrent jobs)")
    hdr = (f"  {'jobs':>5} {'strategy':<14}{'gpu':<9}{'cost':>8} {'GPU-h':>7} "
           f"{'makespan':>9} {'useful':>7}  {'mem':>7} {'time':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for p in shown:
        print(f"  {p['n_chunks']:>5} {p['strategy']:<14}{p['gpu']:<9}"
              f"{p['cost']:>8.2f} {p['gpu_hours']:>7.2f} "
              f"{fmt_span(p['makespan_min']):>9} {p['efficiency']*100:>6.0f}%  "
              f"{p['mem_mb']/1000:>6.0f}G {p['runtime_min']:>5}m")
    if len(front) > len(shown):
        print(f"  ... {len(front)-len(shown)} more frontier plans omitted")

    cheapest = min(plans, key=lambda p: p["cost"])
    fastest = min(plans, key=lambda p: p["makespan_min"])
    value = best_value(plans)
    print(f"\ncheapest   : {cheapest['n_chunks']} job(s) {cheapest['strategy']} on "
          f"{cheapest['gpu']} — cost {cheapest['cost']:.2f}, "
          f"{fmt_span(cheapest['makespan_min'])}, "
          f"{cheapest['efficiency']*100:.0f}% of GPU time doing real work")
    print(f"fastest    : {fastest['n_chunks']} job(s) {fastest['strategy']} on "
          f"{fastest['gpu']} — cost {fastest['cost']:.2f}, "
          f"{fmt_span(fastest['makespan_min'])}, "
          f"{fastest['efficiency']*100:.0f}% useful")
    if cheapest["cost"] > 0:
        print(f"             (fastest costs "
              f"{fastest['cost']/cheapest['cost']:.1f}x the cheapest to finish "
              f"{cheapest['makespan_min']/max(fastest['makespan_min'],1e-9):.1f}x sooner)")
    print(f"best value : {value['n_chunks']} job(s) {value['strategy']} on "
          f"{value['gpu']} — cost {value['cost']:.2f}, "
          f"{fmt_span(value['makespan_min'])}, {value['efficiency']*100:.0f}% useful"
          f"   <- recommended")

    pick = value
    if args.makespan_budget_min:
        ok = [p for p in plans if p["makespan_min"] <= args.makespan_budget_min]
        if ok:
            pick = min(ok, key=lambda p: p["cost"])
            print(f"\nwithin {args.makespan_budget_min:.0f} min budget: "
                  f"{pick['n_chunks']} job(s) {pick['strategy']}, cost {pick['cost']:.2f}")
        else:
            print(f"\nno plan finishes within {args.makespan_budget_min:.0f} min; "
                  f"fastest is {fastest['makespan_min']:.0f} min")
            pick = fastest

    chunk_size = math.ceil(len(lengths) / pick["n_chunks"])
    if m.fit_n_range:
        n_lo, n_hi = m.fit_n_range
        if chunk_size > 2 * n_hi:
            print(f"\n  ! this plan puts {chunk_size} sequences in a job, but the "
                  f"model only ever saw {n_lo}-{n_hi} per job. The host-RAM and "
                  f"runtime numbers below are extrapolated {chunk_size/n_hi:.0f}x "
                  f"beyond the data — treat the memory figure especially as a "
                  f"guess, and verify with one job before running the fleet.")

    print(f"\nrecommended config for {args.stage}:")
    print(f"  {args.stage}.max_files_per_job : {chunk_size}")
    print(f"  slurm.resources.{args.stage}.mem_mb   : {pick['mem_mb']}")
    print(f"  slurm.resources.{args.stage}.runtime  : {pick['runtime_min']}")
    print(f"  partition                     : {GPU_PARTITION.get(pick['gpu'],'')}"
          f"   ({pick['gpu']})")
    if pick["strategy"] == "length-binned":
        print("  note: length-binned beat flat chunking — enable "
              f"{args.stage}.binning to group similar-length sequences")

    if args.out:
        args.out.write_text(yaml.safe_dump({
            "stage": args.stage, "workload": {"n_seqs": len(lengths)},
            "recommended": {k: pick[k] for k in
                            ("n_chunks", "strategy", "gpu", "mem_mb",
                             "runtime_min", "cost", "makespan_min", "efficiency")},
            "chunk_size": chunk_size,
        }, sort_keys=False))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
