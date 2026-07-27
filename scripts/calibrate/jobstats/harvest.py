#!/usr/bin/env python3
"""Right-size SLURM resources from jobs that already ran.

The `stair/` sweep measures scaling deliberately and costs GPU hours. This is
the cheap complement: every production run already leaves behind SLURM jobs, and
`jobstats` knows what they actually used. Harvest that and compare it to what we
asked for.

    jobid + rule   <- .snakemake/log/*.snakemake.log
    GPU util/mem   <- jobstats -j          (the ONLY GPU source on this cluster:
                                            sacct's gres/gpumem is always 0 here)
    host mem, wall <- jobstats, sacct MaxRSS fallback for short jobs
    io_in, cpu_time<- snakemake benchmark TSVs

Two caveats worth knowing before trusting a row:
  * jobstats' sampler has no data for jobs shorter than ~2 min -> `nodes: {}`.
    Those rows fall back to sacct MaxRSS and report gpu=None, not gpu=0.
  * MaxRSS and GPU util are both *sampled*, so a brief peak can be missed.
    Treat recommendations as "no smaller than", never as a tight bound.

Usage:
    python scripts/calibrate/jobstats/harvest.py                  # newest run
    python scripts/calibrate/jobstats/harvest.py --log <path> --json out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# "Job 26 has been submitted with SLURM jobid 30946532 (log: .../rule_run_colabfold_search/4/30946532.log)"
LOG_RE = re.compile(r"SLURM jobid (\d+) \(log: .*?slurm_logs/rule_([A-Za-z_0-9]+)/")

# rule name -> pipeline stage (the config/scaling_models key)
RULE_TO_STAGE = {
    "run_colabfold_search": "msa",
    "run_boltz_predict": "boltz",
    "run_esmc": "esmc",
    "run_esmfold": "esmfold",
    "run_openfold_predict": "openfold",
}

# VRAM per GPU type, smallest first. Mirrors scaling_models.yaml:gpu_specs.
GPU_TIERS = [
    ("a100_mig", 20, "kempner_requeue"),
    ("a100", 40, "kempner"),
    ("a100_80", 80, "kempner"),
    ("h100", 80, "kempner_h100"),
]

MEM_MARGIN = 1.3     # scaling_models.yaml:margins.mem_safety
TIME_MARGIN = 1.5    # scaling_models.yaml:margins.time_safety
VRAM_MARGIN = 1.2
MIN_RUNTIME_MIN = 15
GIB = 2 ** 30

# SM-utilization bands. Above GPU_BOUND_UTIL a stage is compute-bound and must
# keep its fast card; below IDLE_UTIL it is doing nothing on the GPU and only
# needs one that fits.
GPU_BOUND_UTIL = 50.0
IDLE_UTIL = 10.0


def sh(cmd: list[str], timeout: int = 120) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=timeout)
    return p.stdout if p.returncode == 0 else ""


def jobs_from_log(path: Path) -> list[tuple[str, str]]:
    seen, out = set(), []
    for line in path.read_text(errors="replace").splitlines():
        m = LOG_RE.search(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append((m.group(1), m.group(2)))
    return out


def hms_to_sec(v: str) -> float:
    """SLURM durations: '[DD-]HH:MM:SS'. Returns 0 on UNLIMITED/garbage."""
    if not v or v.upper() in ("UNLIMITED", "PARTITION_LIMIT", ""):
        return 0.0
    days = 0
    if "-" in v:
        d, _, v = v.partition("-")
        days = int(d)
    try:
        parts = [float(x) for x in v.split(":")]
    except ValueError:
        return 0.0
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def rss_to_gb(v: str) -> float:
    """sacct MaxRSS: number with a K/M/G suffix."""
    v = (v or "").strip()
    if not v or v[0].isalpha():
        return 0.0
    mult = {"K": 1 / 2**20, "M": 1 / 2**10, "G": 1.0, "T": 1024.0}
    return float(v[:-1]) * mult.get(v[-1].upper(), 1 / 2**30) if v[-1].isalpha() \
        else float(v) / 2**30


def reqmem_to_gb(v: str) -> float:
    """sacct ReqMem: e.g. '250G', '32000M'. Suffix-less values are MB."""
    v = (v or "").strip().rstrip("nc")  # older sacct appends per-node/per-cpu
    if not v:
        return 0.0
    mult = {"K": 1 / 2**20, "M": 1 / 2**10, "G": 1.0, "T": 1024.0}
    if v[-1].isalpha():
        try:
            return float(v[:-1]) * mult.get(v[-1].upper(), 0.0)
        except ValueError:
            return 0.0
    try:
        return float(v) / 1024
    except ValueError:
        return 0.0


def sacct_jobs(jids: list[str]) -> dict:
    """Top-level job records: state, wall, limit, requested mem."""
    out = {}
    txt = sh(["sacct", "-X", "-n", "-P", "-j", ",".join(jids),
              "--format=JobID,State,Elapsed,Timelimit,ReqMem"])
    for line in txt.strip().splitlines():
        f = line.split("|")
        if len(f) >= 5:
            out[f[0]] = dict(state=f[1], wall_s=hms_to_sec(f[2]),
                             limit_s=hms_to_sec(f[3]), reqmem_gb=reqmem_to_gb(f[4]))
    return out


def sacct_maxrss(jids: list[str]) -> dict:
    """Peak MaxRSS across steps — the fallback when jobstats has no samples."""
    out: dict = defaultdict(float)
    txt = sh(["sacct", "-n", "-P", "-j", ",".join(jids),
              "--format=JobID,MaxRSS"])
    for line in txt.strip().splitlines():
        f = line.split("|")
        if len(f) < 2:
            continue
        base = f[0].split(".")[0]
        out[base] = max(out[base], rss_to_gb(f[1]))
    return dict(out)


def jobstats_one(jid: str) -> dict | None:
    txt = sh(["jobstats", "-j", jid])
    if not txt.strip():
        return None
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def benchmarks(output_dir: Path) -> dict:
    """stage -> {'io_in_mb': max, 'cpu_time_s': max} from snakemake benchmark TSVs."""
    out: dict = {}
    bench = output_dir / "benchmarks"
    if not bench.is_dir():
        return out
    for stage_dir in bench.iterdir():
        if not stage_dir.is_dir():
            continue
        io_in, cpu_t, max_rss = [], [], []
        for tsv in stage_dir.glob("*.tsv"):
            lines = tsv.read_text().strip().splitlines()
            if len(lines) < 2:
                continue
            hdr, row = lines[0].split("\t"), lines[1].split("\t")
            rec = dict(zip(hdr, row))
            for key, sink in (("io_in", io_in), ("cpu_time", cpu_t),
                              ("max_rss", max_rss)):
                try:
                    sink.append(float(rec.get(key, 0) or 0))
                except ValueError:
                    pass
        if io_in or cpu_t or max_rss:
            out[stage_dir.name] = dict(
                io_in_mb=max(io_in) if io_in else 0.0,
                cpu_time_s=max(cpu_t) if cpu_t else 0.0,
                # Snakemake writes max_rss in MB. It is a proper high-water mark
                # of the cgroup RSS; jobstats' host-memory number is sampled and
                # on this cluster has undersampled the MSA peak by ~4x. When both
                # exist, trust the larger — under-requesting host RAM OOM-kills.
                max_rss_gb=(max(max_rss) / 1024) if max_rss else 0.0,
            )
    return out


def collect(log: Path) -> list[dict]:
    pairs = jobs_from_log(log)
    if not pairs:
        return []
    jids = [j for j, _ in pairs]
    acct, rss = sacct_jobs(jids), sacct_maxrss(jids)

    rows = []
    for jid, rule in pairs:
        a = acct.get(jid, {})
        js = jobstats_one(jid) or {}
        node = next(iter(js.get("nodes", {}).values()), None)
        wall = js.get("total_time") or a.get("wall_s", 0)

        if node:  # jobstats had samples
            gmem = max(node.get("gpu_used_memory", {}).values(), default=0) / GIB
            gtot = max(node.get("gpu_total_memory", {}).values(), default=0) / GIB
            gutil = max(node.get("gpu_utilization", {}).values(), default=0.0)
            mem_used = node.get("used_memory", 0) / GIB
            mem_alloc = node.get("total_memory", 0) / GIB
            cpus = node.get("cpus", 0) or 0
            cpu_eff = (node.get("total_time", 0) / (wall * cpus) * 100
                       if wall and cpus else 0.0)
            src = "jobstats"
        else:  # too short to sample — degrade honestly, do NOT claim gpu=0
            gmem = gtot = gutil = None
            mem_used = rss.get(jid, 0.0)
            mem_alloc = a.get("reqmem_gb", 0.0)
            cpus = 0
            cpu_eff = None
            src = "sacct"

        rows.append(dict(
            jobid=jid, rule=rule, stage=RULE_TO_STAGE.get(rule, rule),
            state=a.get("state", "?"), source=src, cpus=cpus,
            mem_used_gb=round(mem_used, 2), mem_alloc_gb=round(mem_alloc, 1),
            gpu_mem_used_gb=None if gmem is None else round(gmem, 2),
            gpu_mem_total_gb=None if gtot is None else round(gtot, 1),
            gpu_util_pct=gutil, cpu_eff_pct=None if cpu_eff is None else round(cpu_eff, 1),
            wall_s=round(wall), limit_s=round(a.get("limit_s", 0)),
        ))
    return rows


def recommend(rows: list[dict], bench: dict) -> dict:
    """Per stage: what we'd ask for next time, and why."""
    by = defaultdict(list)
    for r in rows:
        if r["state"] in ("COMPLETED", "RUNNING"):
            by[r["stage"]].append(r)

    recs = {}
    for stage, rs in sorted(by.items()):
        js_mem = max(r["mem_used_gb"] for r in rs)
        # Host-memory high-water: prefer the benchmark's max_rss over jobstats'
        # sampled figure — the latter has undersampled real peaks (MSA ~4x low).
        bench_rss = bench.get(stage, {}).get("max_rss_gb", 0.0)
        peak_mem = max(js_mem, bench_rss)
        mem_src = "benchmark max_rss" if bench_rss > js_mem else "jobstats"
        wall = max(r["wall_s"] for r in rs)
        vram = [r["gpu_mem_used_gb"] for r in rs if r["gpu_mem_used_gb"] is not None]
        utils = [r["gpu_util_pct"] for r in rs if r["gpu_util_pct"] is not None]
        peak_vram = max(vram) if vram else None
        peak_util = max(utils) if utils else None

        mem_mb = max(4000, int(peak_mem * MEM_MARGIN * 1024 / 1000 + 0.999) * 1000)
        runtime = max(MIN_RUNTIME_MIN, int(wall * TIME_MARGIN / 60 + 0.999))

        # GPU choice is NOT a VRAM-fitting problem. A stage pinned at high SM is
        # compute-bound: shrinking it to the smallest card that *fits* would just
        # make it slower for longer. Only downgrade what is demonstrably idle.
        gpu, gpu_reason = None, None
        if peak_vram is not None:
            need = peak_vram * VRAM_MARGIN
            fits = next((n for n, gb, _ in GPU_TIERS if gb >= need), "h100")
            if peak_util is not None and peak_util >= GPU_BOUND_UTIL:
                gpu = "keep current"
                gpu_reason = (f"{peak_util:.0f}% SM — compute-bound, keep the "
                              f"fastest card even though {peak_vram:.1f} GB VRAM "
                              f"would fit on {fits}")
            elif peak_util is not None and peak_util < IDLE_UTIL:
                gpu = fits
                gpu_reason = (f"{peak_util:.1f}% SM and {peak_vram:.1f} GB VRAM — "
                              f"not GPU-bound, {fits} is enough")
            else:
                gpu = fits
                gpu_reason = (f"{peak_util:.0f}% SM, {peak_vram:.1f} GB VRAM — "
                              f"partially GPU-bound; {fits} fits, verify runtime "
                              f"before committing")

        notes = []
        if gpu_reason:
            notes.append(gpu_reason)
        if peak_util is not None and peak_util < IDLE_UTIL:
            notes.append(
                f"GPU essentially idle ({peak_util:.1f}% SM) — should not hold an H100.")
        if peak_util is None:
            notes.append(
                "jobs too short for jobstats to sample (<~2 min) — no GPU data; "
                "memory is sacct MaxRSS and may miss a peak.")
        b = bench.get(stage)
        if b and b["io_in_mb"] > 5000:
            notes.append(
                f"reads {b['io_in_mb']/1024:.1f} GB per job — if that is model "
                f"weights it is a fixed per-job cost that fewer/larger jobs "
                f"amortize; if it is a database scan it scales with the data, "
                f"not the job count.")
        if bench_rss and js_mem and bench_rss / max(js_mem, .01) > 1.5:
            notes.append(
                f"host mem: benchmark caught {bench_rss:.0f} GB but jobstats only "
                f"sampled {js_mem:.1f} GB — sizing off the benchmark ({mem_src}); "
                f"jobstats alone would have under-requested and OOM-killed.")
        alloc = max(r["mem_alloc_gb"] for r in rs)
        if alloc and peak_mem and alloc / max(peak_mem, .01) > 3:
            notes.append(
                f"asked {alloc:.0f} GB, peaked {peak_mem:.1f} GB "
                f"({alloc/peak_mem:.1f}x over).")
        lim = max(r["limit_s"] for r in rs)
        if lim and wall and lim / wall > 3:
            notes.append(
                f"time limit {lim/60:.0f} min vs {wall/60:.1f} min actual "
                f"({lim/wall:.1f}x) — a tighter limit backfills sooner.")

        recs[stage] = dict(
            n_jobs=len(rs), peak_mem_gb=round(peak_mem, 2), mem_source=mem_src,
            peak_vram_gb=peak_vram, peak_gpu_util_pct=peak_util,
            max_wall_min=round(wall / 60, 1),
            recommend=dict(mem_mb=mem_mb, runtime=runtime, gpu=gpu),
            notes=notes,
        )
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, help="snakemake log (default: newest)")
    ap.add_argument("--output-dir", type=Path,
                    help="run output dir, for benchmark TSVs")
    ap.add_argument("--json", type=Path, help="write raw rows + recs here")
    args = ap.parse_args()

    log = args.log
    if log is None:
        logs = sorted(glob.glob(".snakemake/log/*.snakemake.log"))
        if not logs:
            print("no .snakemake/log/*.snakemake.log; pass --log", file=sys.stderr)
            return 1
        log = Path(logs[-1])

    rows = collect(log)
    if not rows:
        print(f"no SLURM jobs found in {log}", file=sys.stderr)
        return 1

    bench = benchmarks(args.output_dir) if args.output_dir else {}
    recs = recommend(rows, bench)

    print(f"run: {log}   jobs: {len(rows)}\n")
    hdr = (f"{'stage':<10}{'n':>3}  {'host GB used/req':>17}  {'VRAM GB':>9}  "
           f"{'GPU%':>5}  {'wall/limit min':>15}")
    print(hdr)
    print("-" * len(hdr))
    for stage, r in recs.items():
        rs = [x for x in rows if x["stage"] == stage]
        alloc = max(x["mem_alloc_gb"] for x in rs)
        vram = "n/a" if r["peak_vram_gb"] is None else f"{r['peak_vram_gb']:.1f}"
        util = "n/a" if r["peak_gpu_util_pct"] is None else f"{r['peak_gpu_util_pct']:.1f}"
        lim = max(x["limit_s"] for x in rs) / 60
        print(f"{stage:<10}{r['n_jobs']:>3}  {r['peak_mem_gb']:>7.1f}/{alloc:<9.0f}  "
              f"{vram:>9}  {util:>5}  {r['max_wall_min']:>6.1f}/{lim:<8.0f}")

    print("\nrecommended resources")
    print("-" * 21)
    for stage, r in recs.items():
        g = r["recommend"]
        print(f"\n{stage}:  mem_mb: {g['mem_mb']}   runtime: {g['runtime']}"
              + (f"   gpu: {g['gpu']}" if g["gpu"] else ""))
        for n in r["notes"]:
            print(f"    - {n}")

    if args.json:
        args.json.write_text(json.dumps(dict(log=str(log), rows=rows, recs=recs),
                                        indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
