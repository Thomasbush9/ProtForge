#!/usr/bin/env python3
"""
Tidy + summarize the stair benchmark's results.csv.

flock-appended rows land in completion order; this sorts them by
(stage, rung_idx), rewrites results.csv in place (sorted), and prints an
aligned per-stage table so the length -> (wall, infer, vram, ram) trend and
the OOM boundary are readable at a glance.

Usage:
    python scripts/calibrate/stair/collect.py --run-dir calib_stair_h100
    python scripts/calibrate/stair/collect.py --csv path/to/results.csv --no-sort
"""

import argparse
import csv
from pathlib import Path

# Stage display order (model stages after msa); unknown stages sort last.
STAGE_ORDER = {
    "msa": 0, "boltz": 1, "esmfold": 2,
    "esmc_300M": 3, "esmc_600M": 4, "esmc_6B": 5,
}
COLS = ["stage", "variant", "gpu_type", "rung_idx", "seq_len", "msa_depth",
        "status", "wall_total_s", "infer_s", "vram_peak_mib", "host_ram_peak_mib"]


def _num(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        return 1 << 30  # NA / non-numeric sorts last


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, help="Run dir holding results.csv")
    ap.add_argument("--csv", type=Path, help="Explicit results.csv path (overrides --run-dir)")
    ap.add_argument("--no-sort", action="store_true", help="Don't rewrite results.csv sorted")
    args = ap.parse_args()

    csv_path = args.csv or (args.run_dir / "results.csv" if args.run_dir else None)
    if not csv_path or not csv_path.exists():
        print(f"results.csv not found ({csv_path})")
        return 1

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("results.csv is empty")
        return 1

    rows.sort(key=lambda r: (STAGE_ORDER.get(r["stage"], 99), _num(r.get("rung_idx"))))

    if not args.no_sort:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"Sorted {len(rows)} rows -> {csv_path}\n")

    # Per-stage aligned summary.
    by_stage: dict[str, list[dict]] = {}
    for r in rows:
        by_stage.setdefault(r["stage"], []).append(r)

    for stage in sorted(by_stage, key=lambda s: STAGE_ORDER.get(s, 99)):
        srows = by_stage[stage]
        variant = srows[0].get("variant", "")
        gpu = srows[0].get("gpu_type", "")
        print(f"=== {stage}  (variant={variant}, gpu={gpu}, n={len(srows)}) ===")
        print(f"  {'rung':>4} {'len':>5} {'depth':>6} {'status':>8} "
              f"{'wall_s':>8} {'infer_s':>8} {'vram_MiB':>9} {'ram_MiB':>8}")
        n_ok = 0
        oom_at = None
        for r in srows:
            if r["status"] == "ok":
                n_ok += 1
            elif oom_at is None and r["status"] in ("oom", "oom-host"):
                oom_at = r.get("seq_len", "?")
            print(f"  {r.get('rung_idx',''):>4} {r.get('seq_len',''):>5} "
                  f"{r.get('msa_depth',''):>6} {r['status']:>8} "
                  f"{r.get('wall_total_s',''):>8} {r.get('infer_s',''):>8} "
                  f"{r.get('vram_peak_mib',''):>9} {r.get('host_ram_peak_mib',''):>8}")
        msg = f"  -> {n_ok}/{len(srows)} ok"
        if oom_at is not None:
            msg += f"; first OOM at seq_len={oom_at}"
        print(msg + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
