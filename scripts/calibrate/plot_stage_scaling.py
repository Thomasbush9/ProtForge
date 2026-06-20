#!/usr/bin/env python3
"""
Plot per-stage GPU scaling: sequence length vs (GPU memory, inference time).

Reads the stair benchmark's results.csv (one measurement per sequence length per
stage — the clean per-length source, unlike production runs which batch mixed
lengths) and draws ONE combined figure with a dual y-axis:

    x            sequence length L (aa)            [results.csv: seq_len]
    y-left       peak GPU memory (MiB)             [results.csv: vram_peak_mib]
    y-right      inference time (s)                [results.csv: infer_s]

Each stage is one color. Memory is drawn solid (o markers), time dashed
(x markers), so within a color the two metrics are distinguishable. Only
status == "ok" rows are plotted (OOM/error rows carry NA for mem/time).

`infer_s` is the model-call-only time (BENCH_INFER_S) where the runner emits it
(esmc/esmfold); for black-box CLIs (msa/boltz) bench_one.sh falls back to wall,
which is noted in the legend. Memory is reserved VRAM sampled by nvidia-smi
across the (exclusive) node during the job — the --gres sizing number.

Usage:
    python scripts/calibrate/plot_stage_scaling.py --run-dir calib_stair_h100
    python scripts/calibrate/plot_stage_scaling.py --csv path/to/results.csv \
        --out /tmp/stage_scaling.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Stage display order + stable color assignment (unknown stages get later colors).
# SAE is intentionally excluded.
STAGE_ORDER = ["msa", "boltz", "esmfold", "esmc_300M", "esmc_600M", "esmc_6B"]

# Stages whose infer_s is really wall time (no inner BENCH_INFER_S timer), so the
# time curve includes load/IO, not just compute. Flagged in the legend.
WALL_AS_INFER = {"msa", "boltz"}


def _num(v):
    """Parse a numeric cell; return None for NA / blank / non-numeric."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def load_results(csv_path: Path) -> dict[str, list[dict]]:
    """Group ok-status rows by stage, each row carrying L, vram, infer."""
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    by_stage: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        L = _num(r.get("seq_len"))
        vram = _num(r.get("vram_peak_mib"))
        infer = _num(r.get("infer_s"))
        if L is None or (vram is None and infer is None):
            continue
        by_stage.setdefault(r["stage"], []).append(
            {"L": L, "vram": vram, "infer": infer}
        )
    for pts in by_stage.values():
        pts.sort(key=lambda p: p["L"])
    return by_stage


def _ordered_stages(by_stage: dict) -> list[str]:
    known = [s for s in STAGE_ORDER if s in by_stage]
    extra = sorted(s for s in by_stage if s not in STAGE_ORDER)
    return known + extra


def plot(by_stage: dict[str, list[dict]], out_path: Path, gpu: str | None):
    stages = _ordered_stages(by_stage)
    if not stages:
        raise SystemExit("No ok-status rows with seq_len found in results.csv")

    cmap = plt.get_cmap("tab10")
    color = {s: cmap(i % 10) for i, s in enumerate(stages)}

    fig, ax_mem = plt.subplots(figsize=(11, 7), constrained_layout=True)
    ax_time = ax_mem.twinx()

    for s in stages:
        pts = by_stage[s]
        c = color[s]
        time_label = f"{s} time" + (" (wall)" if s in WALL_AS_INFER else "")

        mem_pts = [(p["L"], p["vram"]) for p in pts if p["vram"] is not None]
        if mem_pts:
            xs, ys = zip(*mem_pts)
            ax_mem.plot(xs, ys, color=c, marker="o", linewidth=1.8,
                        label=f"{s} GPU mem")

        time_pts = [(p["L"], p["infer"]) for p in pts if p["infer"] is not None]
        if time_pts:
            xs, ys = zip(*time_pts)
            ax_time.plot(xs, ys, color=c, marker="x", linestyle="--",
                         linewidth=1.4, label=time_label)

    ax_mem.set_xlabel("sequence length L (aa)")
    ax_mem.set_ylabel("peak GPU memory (MiB)  —  solid, o")
    ax_time.set_ylabel("inference time (s)  —  dashed, x")
    ax_mem.grid(alpha=0.3)
    ax_mem.set_ylim(bottom=0)
    ax_time.set_ylim(bottom=0)

    # One merged legend (both axes) outside the plot to survive many stages.
    h1, l1 = ax_mem.get_legend_handles_labels()
    h2, l2 = ax_time.get_legend_handles_labels()
    ax_mem.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2,
                  loc="upper left", framealpha=0.9)

    title = "Per-stage scaling — GPU memory & inference time vs sequence length"
    if gpu:
        title += f"  ({gpu})"
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-dir", type=Path,
                    help="Stair run dir holding results.csv")
    ap.add_argument("--csv", type=Path,
                    help="Explicit results.csv path (overrides --run-dir)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG (default: <run-dir>/stage_scaling.png "
                         "or ./stage_scaling.png)")
    ap.add_argument("--gpu", default=None,
                    help="GPU label for the title (default: from first row)")
    args = ap.parse_args()

    csv_path = args.csv or (args.run_dir / "results.csv" if args.run_dir else None)
    if not csv_path or not csv_path.exists():
        raise SystemExit(f"results.csv not found ({csv_path})")

    by_stage = load_results(csv_path)

    gpu = args.gpu
    if gpu is None:
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                if r.get("gpu_type"):
                    gpu = r["gpu_type"]
                    break

    out = args.out or (
        (args.run_dir / "stage_scaling.png") if args.run_dir
        else Path("stage_scaling.png")
    )
    plot(by_stage, out, gpu)
    n = sum(len(v) for v in by_stage.values())
    print(f"Plotted {n} ok rows across {len(by_stage)} stage(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
