#!/usr/bin/env python3
"""
Fit per-stage scaling coefficients from the stair benchmark and emit a
scaling-model block ready to merge into webapp/scaling_models.yaml.

Source: the stair harness's results.csv (one measurement per sequence length per
stage). Columns used: stage, gpu_type, seq_len, status, infer_s, vram_peak_mib.
Only status == "ok" rows are fit (OOM/error rows carry NA).

For each stage it fits, against sequence length L:
  - GPU memory (vram_peak_mib)  -> mem_mb   = base + per_residue·L  (+ alpha_L2·L²)
  - inference time (infer_s)     -> runtime  = base + alpha_L·L      (+ alpha_L2·L²)

GPU memory is the correct sizing signal for --gres/mem_mb (the earlier version
fit HOST RSS, which over- or under-allocated the GPU). The visual length→(mem,
time) figure lives in plot_stage_scaling.py; this tool's unique job is the
coefficient block that drives binning.py's chunk sizing.

Stage name mapping: results.csv uses esmc_300M/600M/6B; scaling_models.yaml uses
a size-agnostic `esmc`. By default the `esmc` block is emitted from the LARGEST
ESM-C size present (worst-case sizing); override with --esmc-from.

Writes:
  - {output_dir}/scaling_models_{gpu}.yaml   coefficient block to merge
  - {output_dir}/calibration_fit.png         per-stage fit overlay (mem + time)
  - {output_dir}/per_stage_summary.txt       console summary + fit MAE

Usage:
  python scripts/calibrate/fit_and_plot.py --run-dir calib_stair_h100
  python scripts/calibrate/fit_and_plot.py --csv results.csv --gpu h100 \
      --output-dir /tmp/fits --esmc-from esmc_600M
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Stages we fit (SAE intentionally excluded). esmc_* are size variants.
STAGE_ORDER = ["msa", "boltz", "esmfold", "esmc_300M", "esmc_600M", "esmc_6B"]

# Stages whose infer_s is wall time (no inner BENCH_INFER_S timer).
WALL_AS_INFER = {"msa", "boltz"}


def _num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def load_results(csv_path: Path) -> dict[str, list[dict]]:
    """Group ok-status rows by stage with L, vram, infer."""
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    by_stage: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        L = _num(r.get("seq_len"))
        if L is None:
            continue
        by_stage.setdefault(r["stage"], []).append({
            "L": L,
            "vram": _num(r.get("vram_peak_mib")),
            "infer": _num(r.get("infer_s")),
        })
    for pts in by_stage.values():
        pts.sort(key=lambda p: p["L"])
    return by_stage


def fit_lsq(L, y, terms):
    """Least-squares fit y = sum(coef[t] * basis(t)) for t in terms.

    terms subset of {"1","L","L2"}. Returns {term: coef}. Degenerate systems
    (fewer points than terms) fall back to a constant mean so callers always get
    a usable block.
    """
    L = np.asarray(L, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(L) < len(terms):
        return {"1": float(np.mean(y)) if len(y) else 0.0}
    cols = []
    for t in terms:
        if t == "1":
            cols.append(np.ones_like(L))
        elif t == "L":
            cols.append(L)
        elif t == "L2":
            cols.append(L * L)
    A = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return dict(zip(terms, coef))


def predict(coefs, L):
    L = np.asarray(L, dtype=float)
    out = np.zeros_like(L)
    out += coefs.get("1", 0.0)
    out += coefs.get("L", 0.0) * L
    out += coefs.get("L2", 0.0) * L * L
    return out


def _terms_for(stage: str, n_points: int, kind: str) -> list[str]:
    """Pick the basis per stage/metric, conservative when data is thin.

    kind: "time" or "mem".
    """
    if stage == "msa":
        return ["1"]               # DB-bound, ~constant
    if stage == "boltz":
        return ["1", "L", "L2"] if n_points >= 3 else ["1", "L"]
    if stage == "esmfold":
        # O(L²) activations; quadratic only when enough points, else linear.
        return ["1", "L", "L2"] if n_points >= 8 else ["1", "L"]
    # esmc_*: ~linear in L for both compute and memory.
    return ["1", "L"]


def _round_sig(x, sig=3):
    x = float(x)
    if x == 0:
        return 0.0
    from math import floor, log10
    return float(round(x, -int(floor(log10(abs(x)))) + (sig - 1)))


def build_block(stage: str, mem_fit: dict, time_fit: dict) -> dict:
    """Shape a stage's fits into the scaling_models.yaml per-gpu schema."""
    mem_block = {"base": int(round(mem_fit.get("1", 0.0)))}
    if "L" in mem_fit:
        mem_block["per_residue"] = _round_sig(mem_fit["L"], 3)
    if "L2" in mem_fit:
        mem_block["alpha_L2"] = _round_sig(mem_fit["L2"], 3)

    if stage == "msa":
        rt_key = "runtime_sec_per_chunk"
        rt_block = {"base": _round_sig(time_fit.get("1", 0.0), 4),
                    "per_seq": 0, "per_residue": 0}
    else:
        rt_key = "runtime_sec_per_seq"
        rt_block = {"base": _round_sig(time_fit.get("1", 0.0), 3)}
        if "L" in time_fit:
            rt_block["alpha_L"] = _round_sig(time_fit["L"], 3)
        if "L2" in time_fit:
            rt_block["alpha_L2"] = _round_sig(time_fit["L2"], 3)
    return {"mem_mb": mem_block, rt_key: rt_block}


def plot_fits(by_stage, fits, mem_fits, out_path, gpu):
    """Diagnostic overlay: measured points + fit line, mem and time, per stage."""
    stages = [s for s in STAGE_ORDER if s in by_stage] + \
             sorted(s for s in by_stage if s not in STAGE_ORDER)
    n = len(stages)
    if n == 0:
        return
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.4 * nrow),
                             constrained_layout=True, squeeze=False)
    for ax, s in zip(axes.flat, stages):
        pts = by_stage[s]
        L = np.array([p["L"] for p in pts])
        mem = np.array([p["vram"] if p["vram"] is not None else np.nan for p in pts])
        tim = np.array([p["infer"] if p["infer"] is not None else np.nan for p in pts])
        grid = np.linspace(L.min(), L.max(), 100) if len(L) else np.array([])

        ax.scatter(L, mem, color="C2", s=36, label="GPU mem (MiB)")
        if len(grid):
            ax.plot(grid, predict(mem_fits[s], grid), color="C2", lw=1.6)
        ax.set_ylabel("GPU mem (MiB)", color="C2")
        ax.tick_params(axis="y", labelcolor="C2")
        ax.set_ylim(bottom=0)

        axt = ax.twinx()
        axt.scatter(L, tim, color="C1", marker="x", s=36,
                    label="infer (s)" + (" [wall]" if s in WALL_AS_INFER else ""))
        if len(grid):
            axt.plot(grid, predict(fits[s], grid), color="C1", ls="--", lw=1.4)
        axt.set_ylabel("time (s)", color="C1")
        axt.tick_params(axis="y", labelcolor="C1")
        axt.set_ylim(bottom=0)

        ax.set_title(s)
        ax.set_xlabel("sequence length L (aa)")
        ax.grid(alpha=0.3)
    # blank any unused panels
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"Stair fits — GPU memory & inference time vs L ({gpu})",
                 fontsize=13)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-dir", type=Path, help="Stair run dir holding results.csv")
    ap.add_argument("--csv", type=Path, help="Explicit results.csv (overrides --run-dir)")
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/calib_fits"))
    ap.add_argument("--gpu", default=None, help="GPU label (default: from first row)")
    ap.add_argument("--esmc-from", default=None,
                    help="Which esmc_* size to emit as the size-agnostic `esmc` "
                         "block (default: largest present).")
    args = ap.parse_args()

    csv_path = args.csv or (args.run_dir / "results.csv" if args.run_dir else None)
    if not csv_path or not csv_path.exists():
        raise SystemExit(f"results.csv not found ({csv_path})")

    by_stage = load_results(csv_path)
    if not by_stage:
        raise SystemExit("No ok-status rows with seq_len in results.csv")

    gpu = args.gpu
    if gpu is None:
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                if r.get("gpu_type"):
                    gpu = r["gpu_type"]
                    break
    gpu = gpu or "h100"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fits, mem_fits = {}, {}
    for s, pts in by_stage.items():
        Lt = [p["L"] for p in pts if p["infer"] is not None]
        yt = [p["infer"] for p in pts if p["infer"] is not None]
        Lm = [p["L"] for p in pts if p["vram"] is not None]
        ym = [p["vram"] for p in pts if p["vram"] is not None]
        fits[s] = fit_lsq(Lt, yt, _terms_for(s, len(Lt), "time"))
        mem_fits[s] = fit_lsq(Lm, ym, _terms_for(s, len(Lm), "mem"))

    # Build the scaling block. esmc_* collapse into one `esmc` entry.
    esmc_sizes = sorted(s for s in by_stage if s.startswith("esmc_"))
    esmc_pick = args.esmc_from or (esmc_sizes[-1] if esmc_sizes else None)
    if args.esmc_from and args.esmc_from not in by_stage:
        raise SystemExit(f"--esmc-from {args.esmc_from} not in results.csv "
                         f"(have: {esmc_sizes})")

    new_block = {}
    for s in by_stage:
        if s.startswith("esmc_"):
            continue  # handled via the collapsed esmc entry
        new_block[s] = {"per_gpu": {gpu: build_block(s, mem_fits[s], fits[s])}}
    if esmc_pick:
        new_block["esmc"] = {
            "per_gpu": {gpu: build_block("esmc_x", mem_fits[esmc_pick], fits[esmc_pick])}
        }

    out_yaml = args.output_dir / f"scaling_models_{gpu}.yaml"
    out_yaml.write_text(yaml.safe_dump(new_block, sort_keys=False))

    fit_png = args.output_dir / "calibration_fit.png"
    plot_fits(by_stage, fits, mem_fits, fit_png, gpu)

    # ---- Summary ---------------------------------------------------------
    def fmt(fit):
        return " + ".join(
            f"{_round_sig(v,3)}{'' if k=='1' else ('·L' if k=='L' else '·L²')}"
            for k, v in fit.items())

    lines = [f"Stair calibration fits — {gpu}", "=" * 60, ""]
    for s in (s for s in STAGE_ORDER if s in by_stage):
        pts = by_stage[s]
        n_mem = sum(1 for p in pts if p["vram"] is not None)
        n_tim = sum(1 for p in pts if p["infer"] is not None)
        lines.append(f"[{s}]  n={len(pts)}  (mem pts={n_mem}, time pts={n_tim})")
        lines.append(f"  mem_mb (MiB) : {fmt(mem_fits[s])}")
        lbl = "infer_s" + (" [wall]" if s in WALL_AS_INFER else "")
        lines.append(f"  {lbl:<12}: {fmt(fits[s])}")
        used = [(p["L"], p["infer"]) for p in pts if p["infer"] is not None]
        if used:
            res = [y - predict(fits[s], [L])[0] for L, y in used]
            lines.append(f"  time fit MAE : {float(np.mean(np.abs(res))):.2f} s")
        lines.append("")
    if esmc_pick:
        lines.append(f"esmc block emitted from: {esmc_pick}")

    summary = args.output_dir / "per_stage_summary.txt"
    summary.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nWrote:")
    for p in (out_yaml, fit_png, summary):
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
