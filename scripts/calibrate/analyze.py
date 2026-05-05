#!/usr/bin/env python3
"""
Join calibration benchmarks with chunk length stats and compare to the
heuristic coefficients in webapp/scaling_models.yaml.

For each stage:
  1. Walks <CALIB>/run/benchmarks/<stage>/*.tsv (Snakemake's per-rule
     benchmark TSVs: wall time, max RSS).
  2. Joins on chunk_id with <CALIB>/run/<stage>_chunks/chunk_stats.tsv.
  3. Writes a clean per-stage CSV (chunk_id, length, wall_s, host_rss_mb).
  4. Prints a summary comparing measured wall time to the heuristic
     coefficients so we can see where to tune.

CAVEAT — host RSS, not GPU memory:
  Snakemake's benchmark: directive captures host process resources via
  psutil/rusage. The numbers in the max_rss column are PyTorch host-side
  allocations, NOT the GPU activation memory that drives partition routing.
  For real GPU memory you'd need to query torch.cuda.max_memory_allocated()
  or post-process nvidia-smi logs. The runtime numbers are clean.

Usage:
  python scripts/calibrate/analyze.py \\
      --calib_dir /path/to/calib_<ts>/run \\
      --output_dir /tmp/calib_analysis
"""

import argparse
import csv
import re
from pathlib import Path

import yaml

# Map: stage -> (benchmark filename regex with chunk_id capture)
STAGE_BENCH_PATTERN = {
    "msa":     re.compile(r"colabfold_search_(\d+)\.tsv$"),
    "boltz":   re.compile(r"predict_(\d+)_run_\d+\.tsv$"),
    "esm":     re.compile(r"esm_chunk_(\d+)\.tsv$"),
    "esmfold": re.compile(r"esmfold_(\d+)\.tsv$"),  # placeholder; adjust if needed
}

STAGE_CHUNK_DIR = {
    "msa":     "msa_chunks",
    "boltz":   "boltz_chunks",
    "esm":     "esm_chunks",
    "esmfold": "esm_chunks",  # esmfold reads YAMLs, often shares the dir
}


def read_tsv(path: Path) -> dict:
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        return {}
    return rows[0]  # benchmark TSVs only have one row


def read_chunk_stats(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        return {int(r["chunk_id"]): r for r in csv.DictReader(f, delimiter="\t")}


def collect_stage(calib_run: Path, stage: str) -> list[dict]:
    """Return list of dicts with joined per-chunk data for one stage."""
    bench_dir = calib_run / "benchmarks" / stage
    if not bench_dir.is_dir():
        return []

    pattern = STAGE_BENCH_PATTERN[stage]
    chunk_stats = read_chunk_stats(
        calib_run / STAGE_CHUNK_DIR[stage] / "chunk_stats.tsv"
    )

    rows = []
    for tsv in sorted(bench_dir.glob("*.tsv")):
        m = pattern.search(tsv.name)
        if not m:
            continue
        chunk_id = int(m.group(1))
        bench = read_tsv(tsv)
        if not bench:
            continue
        stats = chunk_stats.get(chunk_id, {})
        rows.append({
            "chunk_id":      chunk_id,
            "num_seqs":      int(stats.get("num_seqs", 0)) if stats else 0,
            "mean_len":      float(stats.get("mean_len", 0)) if stats else 0,
            "p95_len":       int(stats.get("p95_len", 0)) if stats else 0,
            "max_len":       int(stats.get("max_len", 0)) if stats else 0,
            "wall_s":        float(bench["s"]),
            "host_rss_mb":   float(bench["max_rss"]),
            "cpu_time_s":    float(bench["cpu_time"]),
        })
    rows.sort(key=lambda r: r["mean_len"])
    return rows


def heuristic_predict(stage: str, gpu: str, length: int,
                      scaling: dict, recycling: int = 10,
                      samples: int = 25) -> tuple[float, float] | None:
    """Return (predicted_wall_s, predicted_mem_mb) per scaling_models.yaml."""
    cfg = scaling.get(stage, {})
    per_gpu = cfg.get("per_gpu", {}).get(gpu, {})
    if not per_gpu:
        return None

    mem_coeffs = per_gpu.get("mem_mb", {})
    pred_mem = (
        mem_coeffs.get("base", 0)
        + mem_coeffs.get("per_residue", 0) * length
        + mem_coeffs.get("alpha_L2", 0) * length * length
    )

    if "runtime_sec_per_seq" in per_gpu:
        rt = per_gpu["runtime_sec_per_seq"]
        pred = (
            rt.get("base", 0)
            + rt.get("alpha_L", 0) * length
            + rt.get("alpha_L2", 0) * length * length
        )
        if stage == "boltz":
            pred *= (recycling * samples) / 250.0
        return pred, pred_mem

    if "runtime_sec_per_chunk" in per_gpu:
        rt = per_gpu["runtime_sec_per_chunk"]
        pred = (
            rt.get("base", 0)
            + rt.get("per_seq", 0) * 1
            + rt.get("per_residue", 0) * length
        )
        return pred, pred_mem

    return None, pred_mem


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fit_quadratic(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Cheap least-squares fit for y = a + b*x + c*x^2 (no numpy). Returns (a, b, c)."""
    if len(xs) < 3:
        return 0.0, 0.0, 0.0
    n = len(xs)
    s0 = float(n)
    s1 = sum(xs)
    s2 = sum(x*x for x in xs)
    s3 = sum(x*x*x for x in xs)
    s4 = sum(x*x*x*x for x in xs)
    sy = sum(ys)
    sxy = sum(x*y for x, y in zip(xs, ys))
    sx2y = sum(x*x*y for x, y in zip(xs, ys))

    # Solve 3x3 normal equations via cramer's rule
    def det3(m):
        return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
              - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
              + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))

    A = [[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]]
    detA = det3(A)
    if abs(detA) < 1e-9:
        return 0.0, 0.0, 0.0
    Aa = [[sy, s1, s2], [sxy, s2, s3], [sx2y, s3, s4]]
    Ab = [[s0, sy, s2], [s1, sxy, s3], [s2, sx2y, s4]]
    Ac = [[s0, s1, sy], [s1, s2, sxy], [s2, s3, sx2y]]
    return det3(Aa)/detA, det3(Ab)/detA, det3(Ac)/detA


def print_stage_report(stage: str, rows: list[dict], scaling: dict, gpu: str):
    if not rows:
        print(f"\n[{stage}] no benchmark data\n")
        return
    print(f"\n[{stage}] {len(rows)} samples on {gpu}")
    print("-" * 80)
    print(f"{'L':>6}  {'wall_s':>8}  {'host_rss_mb':>11}  "
          f"{'pred_wall_s':>11}  {'wall_ratio':>10}  {'pred_mem_mb':>11}")
    for r in rows:
        L = int(r["mean_len"])
        pred = heuristic_predict(stage, gpu, L, scaling)
        pred_wall, pred_mem = pred if pred else (None, None)
        ratio = (r["wall_s"] / pred_wall) if pred_wall else None
        print(f"{L:>6}  {r['wall_s']:>8.1f}  {r['host_rss_mb']:>11.1f}  "
              f"{(pred_wall or 0):>11.1f}  "
              f"{(ratio if ratio else 0):>10.2f}  "
              f"{(pred_mem or 0):>11.0f}")

    # Fit measured wall_s vs L (quadratic) so we know the empirical coefficients
    xs = [r["mean_len"] for r in rows]
    ys = [r["wall_s"]   for r in rows]
    a, b, c = fit_quadratic(xs, ys)
    print(f"\n  Empirical fit:  wall_s ≈ {a:.1f} + {b:.4f}·L + {c:.6f}·L²")
    if stage == "boltz":
        # Boltz heuristic divides by recycling*samples/250 = 1 in our setup
        per_gpu = scaling.get("boltz", {}).get("per_gpu", {}).get(gpu, {})
        rt = per_gpu.get("runtime_sec_per_seq", {})
        print(f"  Heuristic    :  wall_s ≈ {rt.get('base', 0):.1f} "
              f"+ 0·L + {rt.get('alpha_L2', 0):.6f}·L²  (current scaling_models.yaml)")
    elif stage == "esm":
        per_gpu = scaling.get("esm", {}).get("per_gpu", {}).get(gpu, {})
        rt = per_gpu.get("runtime_sec_per_seq", {})
        print(f"  Heuristic    :  wall_s ≈ {rt.get('base', 0):.1f} "
              f"+ {rt.get('alpha_L', 0):.4f}·L + {rt.get('alpha_L2', 0):.6f}·L²")
    elif stage == "msa":
        per_gpu = scaling.get("msa", {}).get("per_gpu", {}).get(gpu, {})
        rt = per_gpu.get("runtime_sec_per_chunk", {})
        print(f"  Heuristic    :  wall_s ≈ {rt.get('base', 0):.1f} "
              f"+ {rt.get('per_seq', 0):.4f}·1 + {rt.get('per_residue', 0):.4f}·L")
    elif stage == "esmfold":
        per_gpu = scaling.get("esmfold", {}).get("per_gpu", {}).get(gpu, {})
        rt = per_gpu.get("runtime_sec_per_seq", {})
        print(f"  Heuristic    :  wall_s ≈ {rt.get('base', 0):.1f} "
              f"+ 0·L + {rt.get('alpha_L2', 0):.6f}·L²")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calib_dir", type=Path, required=True,
                        help="Path to calibration <output>/run/ directory")
    parser.add_argument("--output_dir", type=Path, default=Path("calib_analysis"),
                        help="Where to write per-stage joined CSVs")
    parser.add_argument("--gpu", default="h100",
                        help="GPU type the sweep ran on (matches scaling_models.yaml keys)")
    parser.add_argument("--scaling_models", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent
                                / "webapp" / "scaling_models.yaml",
                        help="Path to webapp/scaling_models.yaml for heuristic comparison")
    args = parser.parse_args()

    if not args.calib_dir.is_dir():
        raise SystemExit(f"calib_dir not found: {args.calib_dir}")

    scaling = yaml.safe_load(args.scaling_models.read_text())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Calibration data: {args.calib_dir}")
    print(f"GPU type        : {args.gpu}")
    print(f"Heuristics from : {args.scaling_models}")

    for stage in ("msa", "boltz", "esm", "esmfold"):
        rows = collect_stage(args.calib_dir, stage)
        if rows:
            csv_path = args.output_dir / f"{stage}_calibration.csv"
            write_csv(csv_path, rows)
            print(f"\nWrote {csv_path} ({len(rows)} rows)")
        print_stage_report(stage, rows, scaling, args.gpu)

    print("\n" + "=" * 80)
    print("CAVEAT: host_rss_mb is the snakemake-captured host RSS, not the")
    print("GPU activation memory that drives partition routing. To capture")
    print("GPU mem we need a torch.cuda.max_memory_allocated() wrapper or")
    print("nvidia-smi polling — not in scope yet.")


if __name__ == "__main__":
    main()
