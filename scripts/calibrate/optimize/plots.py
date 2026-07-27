#!/usr/bin/env python3
"""Curves for the cost model: validation first, then the optimization trade-off.

Three figures:

  validation.png  — predicted vs observed job wall time, per stage. This is the
                    one to look at first: if the points do not sit on the
                    diagonal, nothing downstream should be trusted.
  scaling.png     — the measured length curves (VRAM and per-sequence time)
                    with the fitted lines over them.
  tradeoff.png    — cost and makespan vs number of chunks, per stage, with the
                    overhead share shaded. This is the picture that answers
                    "how many jobs should I split into?": where the cost curve
                    is flat, split more; where it climbs, do not.

Usage:
    python -m scripts.calibrate.optimize.plots --obs observations.csv \
        --model cost_model.yaml [--stair results.csv ...] --out-dir figures/
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan import Model, chunks_balanced, evaluate  # noqa: E402

STAGE_COLORS = {"msa": "C0", "boltz": "C1", "esmc": "C2",
                "esmfold": "C3", "openfold": "C4"}


def _f(v, d=0.0):
    try:
        return float(v) if v not in (None, "") else d
    except (TypeError, ValueError):
        return d


def load_obs(path: Path) -> list[dict]:
    with path.open() as f:
        return [r for r in csv.DictReader(f) if _f(r.get("wall_s")) > 0]


def load_stair(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        if not p.exists():
            continue
        with p.open() as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok":
                    out.append(r)
    return out


def fig_validation(obs, model, out: Path):
    """Predicted vs observed wall time. The honest check on the whole model."""
    stages = [s for s in model if any(r["stage"] == s for r in obs)]
    if not stages:
        return
    ncol = min(3, len(stages))
    nrow = math.ceil(len(stages) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.8 * nrow),
                             squeeze=False, constrained_layout=True)
    for ax, stage in zip(axes.flat, stages):
        m = Model(model[stage])
        rows = [r for r in obs if r["stage"] == stage]
        pred, act, util = [], [], []
        for r in rows:
            N = int(_f(r["n_seqs"]))
            if N <= 0:
                continue
            # Reconstruct Σf(L) from the stored moments, exactly as fitted.
            work = m.scale * (m.a * N + m.b * _f(r["total_residues"])
                              + m.c * _f(r["sum_len2"]))
            pred.append(m.overhead + work)
            act.append(_f(r["wall_s"]))
            # None means jobstats had no samples (job under ~2 min) — that is
            # "unknown", NOT "idle". Conflating them mislabels every short job.
            raw = r.get("gpu_util_pct")
            util.append(float(raw) if raw not in (None, "") else np.nan)
        if not pred:
            ax.axis("off")
            continue
        pred, act, util = np.array(pred), np.array(act), np.array(util)
        unknown = np.isnan(util)
        idle = ~unknown & (util < 2)
        worked = ~unknown & ~idle

        color = STAGE_COLORS.get(stage, "C7")
        if worked.any():
            ax.scatter(pred[worked], act[worked], s=26, alpha=.65, color=color,
                       label="did work")
        if unknown.any():
            ax.scatter(pred[unknown], act[unknown], s=26, alpha=.65, color=color,
                       marker="s", label="no GPU data (job too short)")
        if idle.any():
            ax.scatter(pred[idle], act[idle], s=30, alpha=.8, facecolors="none",
                       edgecolors="crimson", label="idle GPU (no work done)")
        hi = max(pred.max(), act.max()) * 1.1
        ax.plot([0, hi], [0, hi], "k--", lw=1, label="perfect")
        scored = worked | unknown
        mape = (float(np.median(np.abs(act[scored] - pred[scored]) /
                                np.maximum(act[scored], 1))) * 100
                if scored.any() else float("nan"))
        ax.set_title(f"{stage}  (median error {mape:.0f}%)", fontsize=10)
        ax.set_xlabel("predicted wall (s)")
        ax.set_ylabel("observed wall (s)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    for ax in axes.flat[len(stages):]:
        ax.axis("off")
    fig.suptitle("Model validation — predicted vs observed job time", fontsize=13)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_scaling(stair, model, out: Path):
    """Measured length curves with the fits drawn over them."""
    stages = sorted({r["stage"].split("_")[0] for r in stair})
    stages = [s for s in stages if s in model]
    if not stages:
        return
    fig, axes = plt.subplots(1, len(stages), figsize=(4.4 * len(stages), 3.8),
                             squeeze=False, constrained_layout=True)
    for ax, stage in zip(axes.flat, stages):
        rows = [r for r in stair if r["stage"].split("_")[0] == stage]
        L = np.array([_f(r["seq_len"]) for r in rows])
        vram = np.array([_f(r.get("vram_peak_mib")) / 1024 for r in rows])
        infer = np.array([_f(r.get("infer_s")) for r in rows])
        order = np.argsort(L)
        L, vram, infer = L[order], vram[order], infer[order]
        m = Model(model[stage])

        ax.scatter(L, vram, s=30, color="C2", label="VRAM measured")
        grid = np.linspace(L.min(), L.max(), 200)
        ax.plot(grid, [m.job_vram([g]) for g in grid], color="C2", lw=1.5,
                label="VRAM fit")
        ax.set_ylabel("GPU memory (GB)", color="C2")
        ax.tick_params(axis="y", labelcolor="C2")
        ax.set_ylim(bottom=0)

        axt = ax.twinx()
        axt.scatter(L, infer, s=30, marker="x", color="C1",
                    label="inference measured")
        axt.plot(grid, [m.seq_cost(g) for g in grid], color="C1", ls="--", lw=1.4,
                 label="per-seq fit")
        axt.set_ylabel("per-sequence time (s)", color="C1")
        axt.tick_params(axis="y", labelcolor="C1")
        axt.set_ylim(bottom=0)

        ax.set_xlabel("sequence length (aa)")
        ax.set_title(stage, fontsize=11)
        ax.grid(alpha=.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axt.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    fig.suptitle("Measured scaling vs sequence length, with fits", fontsize=13)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_tradeoff(model, obs, out: Path, concurrency: int = 10):
    """Cost and makespan vs chunk count — the picture that sets chunk size."""
    stages = [s for s in model if model[s].get("time")]
    if not stages:
        return
    ncol = min(3, len(stages))
    nrow = math.ceil(len(stages) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow),
                             squeeze=False, constrained_layout=True)

    for ax, stage in zip(axes.flat, stages):
        m = Model(model[stage])
        rows = [r for r in obs if r["stage"] == stage]
        if rows:
            n_tot = int(sum(_f(r["n_seqs"]) for r in rows[:max(1, len(rows))]))
            mean_len = float(np.mean([_f(r["mean_len"]) for r in rows])) or 300
        else:
            n_tot, mean_len = 100, 300
        n_tot = max(n_tot, 10)
        lengths = [int(mean_len)] * n_tot

        ncs, costs, spans, effs = [], [], [], []
        for nc in range(1, min(n_tot, 40) + 1):
            ev = evaluate(chunks_balanced(lengths, nc, m.seq_cost), m,
                          concurrency, ["a100_mig", "a100", "a100_80", "h100"],
                          750.0, 10 ** 9)
            if ev:
                ncs.append(nc)
                costs.append(ev["gpu_hours"])
                spans.append(ev["makespan_min"])
                effs.append(ev["efficiency"] * 100)
        if not ncs:
            ax.axis("off")
            continue

        ax.plot(ncs, costs, color="C3", lw=1.8, label="total GPU-hours")
        base = costs[0]
        ax.fill_between(ncs, base, costs, color="C3", alpha=.15,
                        label="overhead added by splitting")
        ax.set_xlabel("number of chunks (jobs)")
        ax.set_ylabel("total GPU-hours", color="C3")
        ax.tick_params(axis="y", labelcolor="C3")
        ax.set_ylim(bottom=0)

        axt = ax.twinx()
        axt.plot(ncs, spans, color="C0", ls="--", lw=1.6, label="makespan (min)")
        axt.set_ylabel("makespan (min)", color="C0")
        axt.tick_params(axis="y", labelcolor="C0")
        axt.set_ylim(bottom=0)

        growth = costs[-1] / costs[0] if costs[0] > 0 else 1.0
        ax.set_title(f"{stage} — n={n_tot}, overhead {m.overhead:.0f}s/job\n"
                     f"cost x{growth:.1f} from 1 to {ncs[-1]} jobs "
                     f"(useful {effs[0]:.0f}% -> {effs[-1]:.0f}%)", fontsize=9)
        ax.grid(alpha=.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axt.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")

    for ax in axes.flat[len(stages):]:
        ax.axis("off")
    fig.suptitle(f"Cost vs parallelism ({concurrency} concurrent jobs) — "
                 f"flat red curve means splitting is free", fontsize=13)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--stair", type=Path, action="append", default=[])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-concurrent", type=int, default=10)
    args = ap.parse_args()

    obs = load_obs(args.obs)
    model = yaml.safe_load(args.model.read_text())
    stair = load_stair(args.stair)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig_validation(obs, model, args.out_dir / "validation.png")
    if stair:
        fig_scaling(stair, model, args.out_dir / "scaling.png")
    fig_tradeoff(model, obs, args.out_dir / "tradeoff.png", args.max_concurrent)

    made = sorted(p.name for p in args.out_dir.glob("*.png"))
    print(f"wrote {len(made)} figure(s) to {args.out_dir}: {', '.join(made)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
