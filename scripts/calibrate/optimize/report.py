#!/usr/bin/env python3
"""Per-job efficiency report: SM utilization, RAM, and time split two ways.

Reports the four numbers that decide whether a chunking choice was right:

  SM utilization  what fraction of the GPU's compute was actually used. Low SM
                  on an expensive card is the clearest waste signal there is.
  RAM             peak used vs requested. Over-requesting blocks other jobs from
                  the node; under-requesting OOM-kills.
  total time      per job, and against what was requested.
  time/sequence   wall / n_seqs. This is the metric that exposes fixed overhead:
                  if it FALLS as chunks get bigger, the job is overhead-bound and
                  should be batched harder. If it stays flat, cost is genuinely
                  per-sequence and chunk size is free to pick on other grounds.

`--compare` puts two runs side by side and reports the implied marginal cost of
one more sequence, which is the parameter a single-chunk-size run cannot measure.

Usage:
    python -m scripts.calibrate.optimize.report --obs observations.csv
    python -m scripts.calibrate.optimize.report --obs observations.csv \
        --stage msa --compare
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def _f(v, d=0.0):
    try:
        return float(v) if v not in (None, "") else d
    except (TypeError, ValueError):
        return d


def load(path: Path, stage: str | None) -> list[dict]:
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if _f(r.get("wall_s")) > 0]
    return [r for r in rows if not stage or r["stage"] == stage]


def group_key(r: dict) -> tuple:
    return (r["stage"], r.get("log", r.get("run", "")), int(_f(r["n_seqs"])))


def summarize(rows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[group_key(r)].append(r)
    out = []
    for (stage, run, n), rs in sorted(by.items()):
        walls = [_f(r["wall_s"]) for r in rs]
        rams = [_f(r["host_gb"]) for r in rs]
        reqs = [_f(r["req_mem_gb"]) for r in rs]
        reqt = [_f(r["req_runtime_min"]) for r in rs]
        sms = [_f(r["gpu_util_pct"], None) for r in rs
               if r.get("gpu_util_pct") not in (None, "")]
        med_wall = sorted(walls)[len(walls) // 2]
        out.append({
            "stage": stage, "run": run, "n_seqs": n, "jobs": len(rs),
            "wall_med": med_wall,
            "wall_min": min(walls), "wall_max": max(walls),
            "per_seq": med_wall / n if n else 0.0,
            "ram_peak": max(rams) if rams else 0.0,
            "ram_req": max(reqs) if reqs else 0.0,
            "runtime_req_min": max(reqt) if reqt else 0.0,
            "sm_med": (sorted(sms)[len(sms) // 2] if sms else None),
            "sm_max": (max(sms) if sms else None),
        })
    return out


def print_table(summ: list[dict]) -> None:
    hdr = (f"{'stage':<9}{'N/job':>6}{'jobs':>5}  {'SM %':>11}  "
           f"{'RAM used/req':>16}  {'total time':>18}  {'time/seq':>9}")
    print(hdr)
    print("-" * len(hdr))
    for s in summ:
        sm = "n/a" if s["sm_med"] is None else f"{s['sm_med']:.0f} (max {s['sm_max']:.0f})"
        ramfrac = (s["ram_peak"] / s["ram_req"] * 100) if s["ram_req"] else 0
        spread = ("" if s["jobs"] < 2 else
                  f" [{s['wall_min']/60:.0f}-{s['wall_max']/60:.0f}]")
        reqm = f"/{s['runtime_req_min']:.0f}m req" if s["runtime_req_min"] else ""
        print(f"{s['stage']:<9}{s['n_seqs']:>6}{s['jobs']:>5}  {sm:>11}  "
              f"{s['ram_peak']:>6.0f}/{s['ram_req']:<4.0f} {ramfrac:>3.0f}%  "
              f"{s['wall_med']/60:>7.1f}m{spread:<8}{reqm:<9}  "
              f"{s['per_seq']:>7.1f}s")


def compare(summ: list[dict]) -> None:
    """Marginal cost of a sequence, from two runs with different chunk sizes."""
    by_stage = defaultdict(list)
    for s in summ:
        by_stage[s["stage"]].append(s)
    for stage, group in sorted(by_stage.items()):
        pts = sorted({(g["n_seqs"], g["wall_med"]) for g in group})
        if len(pts) < 2:
            continue
        print(f"\n{stage}: marginal cost of one more sequence")
        (n1, w1), (n2, w2) = pts[0], pts[-1]
        slope = (w2 - w1) / (n2 - n1)
        fixed = w1 - slope * n1
        print(f"  N={n1} -> {w1/60:.1f} min      N={n2} -> {w2/60:.1f} min")
        print(f"  implied per-sequence cost : {slope:+.1f} s/seq")
        print(f"  implied fixed cost per job: {fixed/60:.1f} min")
        if slope <= 0.5:
            print("  => cost is essentially per-JOB, not per-sequence: batch harder, "
                  "fewer chunks is strictly cheaper.")
        else:
            n_star = fixed / slope
            print(f"  => real per-sequence cost. Overhead stops dominating past "
                  f"~{n_star:.0f} seqs/job; beyond that, splitting is nearly free.")
        # Per-sequence time falling with N is the overhead fingerprint.
        p1, p2 = w1 / n1, w2 / n2
        print(f"  time/seq: {p1:.1f}s at N={n1}  ->  {p2:.1f}s at N={n2}  "
              f"({(1-p2/p1)*100:+.0f}% change)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", type=Path, required=True)
    ap.add_argument("--stage")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    rows = load(args.obs, args.stage)
    if not rows:
        print("no matching jobs", file=sys.stderr)
        return 1
    summ = summarize(rows)
    print_table(summ)
    if args.compare:
        compare(summ)
    return 0


if __name__ == "__main__":
    sys.exit(main())
