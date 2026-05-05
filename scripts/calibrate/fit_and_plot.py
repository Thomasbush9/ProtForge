#!/usr/bin/env python3
"""
Fit per-stage scaling coefficients from a calibration sweep and produce plots.

Reads <calib>/run/benchmarks/<stage>/*.tsv joined with <stage>_chunks/chunk_stats.tsv,
fits runtime~L (and host_rss~L), drops obvious outliers (cold-cache model loads
on ESMFold, identified by io_in MB), and writes:

  - {output_dir}/calibration_fits.png       4-panel wall_s vs L
  - {output_dir}/calibration_memory.png     4-panel host_rss vs L
  - {output_dir}/scaling_models_h100.yaml   coefficient block ready to merge
  - {output_dir}/per_stage_summary.txt      console summary

The intent: produce ready-to-drop coefficients for webapp/scaling_models.yaml's
h100 block, plus plots that explain the calibration to a reader.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


STAGE_BENCH = {
    "msa":     ("benchmarks/msa",     "msa_chunks",     "colabfold_search_{cid}.tsv"),
    "boltz":   ("benchmarks/boltz",   "boltz_chunks",   "predict_{cid}_run_0.tsv"),
    "esm":     ("benchmarks/esm",     "esm_chunks",     "esm_chunk_{cid}.tsv"),
    "esmfold": ("benchmarks/esmfold", "esmfold_chunks", "esmfold_chunk_{cid}.tsv"),
}


def _chunk_seq_id(run_dir: Path, stage: str, cid: int) -> str | None:
    """Read first sequence path from <stage>_chunks/id_<cid>.txt → return seq stem."""
    chunk_subdir = STAGE_BENCH[stage][1]
    p = run_dir / chunk_subdir / f"id_{cid}.txt"
    if not p.exists():
        return None
    line = p.read_text().splitlines()[0].strip()
    if not line:
        return None
    return Path(line).stem


def _esmfold_succeeded(run_dir: Path, seq_id: str) -> bool:
    """ESMFold output check — silent OOMs leave no structure.pdb."""
    return (run_dir / "sequences" / seq_id / "esmfold" / "structure.pdb").exists()


def load_stage(run_dir: Path, stage: str):
    bench_subdir, chunk_subdir, _ = STAGE_BENCH[stage]
    bench_dir = run_dir / bench_subdir
    chunk_stats_path = run_dir / chunk_subdir / "chunk_stats.tsv"

    chunk_stats = {}
    with open(chunk_stats_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            chunk_stats[int(r["chunk_id"])] = r

    rows = []
    for tsv in sorted(bench_dir.glob("*.tsv")):
        stem = tsv.stem
        cid = int(stem.split("_")[-1] if not stem.endswith("_run_0")
                  else stem.split("_")[-3])
        s = chunk_stats[cid]
        with open(tsv) as f:
            b = next(csv.DictReader(f, delimiter="\t"))
        seq_id = _chunk_seq_id(run_dir, stage, cid)
        succeeded = True
        if stage == "esmfold" and seq_id is not None:
            succeeded = _esmfold_succeeded(run_dir, seq_id)
        rows.append({
            "chunk_id":    cid,
            "seq_id":      seq_id,
            "L":           float(s["mean_len"]),
            "wall_s":      float(b["s"]),
            "rss_mb":      float(b["max_rss"]),
            "io_in_mb":    float(b["io_in"]),
            "cpu_time_s":  float(b["cpu_time"]),
            "succeeded":   succeeded,
        })
    rows.sort(key=lambda r: r["L"])
    return rows


def fit_lsq(L, y, terms):
    """Least-squares fit y = sum(coef[t] * basis(t)) for t in terms.
    terms: subset of {'1','L','L2'}. Returns dict {term: coef}.
    """
    L = np.asarray(L, dtype=float)
    y = np.asarray(y, dtype=float)
    cols = []
    for t in terms:
        if t == "1":   cols.append(np.ones_like(L))
        elif t == "L": cols.append(L)
        elif t == "L2": cols.append(L * L)
    X = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(zip(terms, coef))


def predict(coefs, L):
    L = np.asarray(L, dtype=float)
    out = np.zeros_like(L)
    if "1"  in coefs: out += coefs["1"]
    if "L"  in coefs: out += coefs["L"]  * L
    if "L2" in coefs: out += coefs["L2"] * L * L
    return out


def heuristic_eval(per_gpu_block, L, kind):
    """kind: 'time' or 'mem'."""
    if kind == "time":
        if "runtime_sec_per_seq" in per_gpu_block:
            rt = per_gpu_block["runtime_sec_per_seq"]
            return (rt.get("base", 0)
                    + rt.get("alpha_L", 0) * L
                    + rt.get("alpha_L2", 0) * L * L)
        if "runtime_sec_per_chunk" in per_gpu_block:
            rt = per_gpu_block["runtime_sec_per_chunk"]
            return (rt.get("base", 0)
                    + rt.get("per_seq", 0) * 1
                    + rt.get("per_residue", 0) * L)
    if kind == "mem":
        m = per_gpu_block.get("mem_mb", {})
        return (m.get("base", 0)
                + m.get("per_residue", 0) * L
                + m.get("alpha_L2", 0) * L * L)
    return None


def filter_for_fit(rows, stage, io_threshold_mb=200):
    """Drop cold-cache loads (high io_in) and silent failures (esmfold only).

    Returns (keep, drop_cold, drop_failed).
    """
    keep, drop_cold, drop_failed = [], [], []
    for r in rows:
        if not r.get("succeeded", True):
            drop_failed.append(r)
            continue
        if stage in ("esm", "esmfold") and r["io_in_mb"] > io_threshold_mb:
            drop_cold.append(r)
            continue
        keep.append(r)
    return keep, drop_cold, drop_failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib_dir", type=Path, required=True,
                    help="<output>/run/ from calibrate.sh")
    ap.add_argument("--output_dir", type=Path, default=Path("/tmp/calib_fits"))
    ap.add_argument("--gpu", default="h100")
    ap.add_argument("--scaling_models", type=Path,
                    default=Path("webapp/scaling_models.yaml"))
    ap.add_argument("--io_threshold_mb", type=float, default=200,
                    help="ESMFold: chunks with io_in above this are treated as "
                         "cold-cache loads and excluded from the fit.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scaling = yaml.safe_load(Path(args.scaling_models).read_text())

    stages = ("msa", "boltz", "esm", "esmfold")
    data = {s: load_stage(args.calib_dir, s) for s in stages}

    fits = {}
    mem_fits = {}
    excluded_cold = {}
    excluded_failed = {}

    for s in stages:
        rows = data[s]
        rows_fit, drop_cold, drop_failed = filter_for_fit(rows, s, args.io_threshold_mb)
        excluded_cold[s] = drop_cold
        excluded_failed[s] = drop_failed

        L = [r["L"] for r in rows_fit]
        y = [r["wall_s"] for r in rows_fit]
        m = [r["rss_mb"] for r in rows_fit]

        if s == "msa":
            terms = ["1"]
        elif s == "esm":
            terms = ["1", "L"]
        elif s == "esmfold":
            # After cold + OOM filters n is small; quadratic overfits → linear
            terms = ["1", "L"] if len(L) < 8 else ["1", "L", "L2"]
        else:  # boltz
            terms = ["1", "L", "L2"]

        fits[s] = fit_lsq(L, y, terms) if len(L) >= len(terms) else {t: 0.0 for t in terms}
        mem_terms = ["1"] if s in ("boltz", "esm") else ["1", "L"]
        mem_fits[s] = fit_lsq(L, m, mem_terms) if len(L) >= len(mem_terms) else {t: 0.0 for t in mem_terms}

    # ---- Plots: runtime ---------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    titles = {
        "msa":     "MSA (ColabFold/MMseq2)  — DB-bound, ~constant",
        "boltz":   "Boltz  — O(L) + O(L²) attention",
        "esm":     "ESM  — embeddings, ~linear in L",
        "esmfold": "ESMFold  — outliers = cold-cache model loads",
    }
    for ax, s in zip(axes.flat, stages):
        rows = data[s]
        L_all = np.array([r["L"] for r in rows])
        y_all = np.array([r["wall_s"] for r in rows])
        cold_set = excluded_cold[s]
        fail_set = excluded_failed[s]
        is_cold = np.array([r in cold_set for r in rows])
        is_fail = np.array([r in fail_set for r in rows])
        is_used = ~(is_cold | is_fail)

        ax.scatter(L_all[is_used], y_all[is_used], s=40, color="C0",
                   label="measured (used)")
        if is_cold.any():
            ax.scatter(L_all[is_cold], y_all[is_cold], s=40, color="C3",
                       marker="x", label="cold-cache (excluded)")
        if is_fail.any():
            ax.scatter(L_all[is_fail], y_all[is_fail], s=60, color="grey",
                       marker="s", facecolors="none", linewidth=1.5,
                       label="silent OOM (no PDB)")

        L_grid = np.linspace(50, max(L_all) * 1.05, 200)
        ax.plot(L_grid, predict(fits[s], L_grid), color="C0",
                label="empirical fit", linewidth=2)
        per_gpu = scaling.get(s, {}).get("per_gpu", {}).get(args.gpu, {})
        if per_gpu:
            yh = heuristic_eval(per_gpu, L_grid, "time")
            if yh is not None and not np.allclose(yh, 0):
                ax.plot(L_grid, yh, color="C1", linestyle="--",
                        label="current heuristic")

        ax.set_title(titles[s])
        ax.set_xlabel("sequence length L (aa)")
        ax.set_ylabel("wall time (s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if s == "esmfold":
            ax.set_yscale("symlog", linthresh=10)

    fig.suptitle(f"H100 calibration — runtime vs sequence length", fontsize=13)
    runtime_png = args.output_dir / "calibration_fits.png"
    fig.savefig(runtime_png, dpi=150)
    plt.close(fig)

    # ---- Plots: memory ----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, s in zip(axes.flat, stages):
        rows = data[s]
        L_all = np.array([r["L"] for r in rows])
        m_all = np.array([r["rss_mb"] for r in rows])
        cold_set = excluded_cold[s]
        fail_set = excluded_failed[s]
        is_used = np.array([r not in cold_set and r not in fail_set for r in rows])
        ax.scatter(L_all[is_used], m_all[is_used], s=40, color="C2",
                   label="measured (host RSS, used)")
        if (~is_used).any():
            ax.scatter(L_all[~is_used], m_all[~is_used], s=40, color="grey",
                       marker="x", alpha=0.6, label="excluded")

        L_grid = np.linspace(50, max(L_all) * 1.05, 200)
        # Empirical fit
        ax.plot(L_grid, predict(mem_fits[s], L_grid), color="C2",
                linewidth=2, label="empirical mem fit")
        # Heuristic
        per_gpu = scaling.get(s, {}).get("per_gpu", {}).get(args.gpu, {})
        if per_gpu:
            mh = heuristic_eval(per_gpu, L_grid, "mem")
            if mh is not None:
                ax.plot(L_grid, mh, color="C1", linestyle="--",
                        label="current heuristic")
        ax.set_title(titles[s])
        ax.set_xlabel("sequence length L (aa)")
        ax.set_ylabel("max host RSS (MB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("H100 calibration — host memory (RSS) vs sequence length",
                 fontsize=13)
    mem_png = args.output_dir / "calibration_memory.png"
    fig.savefig(mem_png, dpi=150)
    plt.close(fig)

    # ---- Plots: optimal chunk size vs L (per stage) ----------------------
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), constrained_layout=True)
    L_grid = np.linspace(100, 1500, 200)
    JOB_OVERHEAD = 30.0  # SLURM scheduling + python startup, seconds
    chunk_rows = []
    for s in stages:
        per_seq = predict(fits[s], L_grid)
        per_seq = np.maximum(per_seq, 1.0)  # floor for sane plotting
        target_min = scaling.get(s, {}).get("target_chunk_runtime_min", 30)
        target_s = target_min * 60.0
        # n_per_chunk such that JOB_OVERHEAD + n*per_seq ≈ target
        n_per_chunk = np.maximum(1, np.floor((target_s - JOB_OVERHEAD) / per_seq))
        max_chunk = scaling.get(s, {}).get("max_chunk_size", 500)
        min_chunk = scaling.get(s, {}).get("min_chunk_size", 1)
        n_per_chunk = np.clip(n_per_chunk, min_chunk, max_chunk)
        ax.plot(L_grid, n_per_chunk, label=f"{s}  (target {target_min} min)",
                linewidth=2)
        for L_pt in (250, 500, 1000, 1500):
            idx = np.argmin(np.abs(L_grid - L_pt))
            chunk_rows.append({"stage": s, "L": int(L_pt),
                               "per_seq_s": float(per_seq[idx]),
                               "n_per_chunk": int(n_per_chunk[idx])})
    ax.set_xlabel("sequence length L (aa)")
    ax.set_ylabel("optimal sequences per chunk")
    ax.set_yscale("log")
    ax.set_title(f"Recommended chunk size vs L (h100, "
                 f"job overhead = {JOB_OVERHEAD:.0f}s)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    chunk_png = args.output_dir / "calibration_chunking.png"
    fig.savefig(chunk_png, dpi=150)
    plt.close(fig)

    # CSV: per-stage chunking recommendations at representative L
    chunk_csv = args.output_dir / "chunking_recommendations.csv"
    with open(chunk_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "L", "per_seq_s", "n_per_chunk"])
        w.writeheader()
        w.writerows(chunk_rows)

    # ---- Coefficient block ------------------------------------------------
    def round_sig(x, sig=3):
        x = float(x)
        if x == 0: return 0.0
        from math import log10, floor
        return float(round(x, -int(floor(log10(abs(x)))) + (sig - 1)))

    def time_block(s, fit):
        if s == "msa":
            # constant base, project into existing per_chunk schema
            return {"base": round_sig(fit.get("1", 0), 4),
                    "per_seq": 0,
                    "per_residue": 0}
        block = {"base": round_sig(fit.get("1", 0), 3)}
        if "L"  in fit: block["alpha_L"]  = round_sig(fit["L"], 3)
        if "L2" in fit: block["alpha_L2"] = round_sig(fit["L2"], 3)
        return block

    new_block = {}
    for s in stages:
        rt = time_block(s, fits[s])
        mem = mem_fits[s]
        mem_block = {"base": int(round(float(mem.get("1", 0))))}
        if "L" in mem: mem_block["per_residue"] = round_sig(mem["L"], 3)
        new_block[s] = {
            "per_gpu": {
                args.gpu: {
                    "mem_mb": mem_block,
                    ("runtime_sec_per_chunk" if s == "msa"
                     else "runtime_sec_per_seq"): rt,
                }
            }
        }

    out_yaml = args.output_dir / f"scaling_models_{args.gpu}.yaml"
    out_yaml.write_text(yaml.safe_dump(new_block, sort_keys=False))

    # ---- Summary text -----------------------------------------------------
    lines = [f"Calibration fits — {args.gpu}", "=" * 60, ""]
    for s in stages:
        rows = data[s]
        cold = excluded_cold[s]
        failed = excluded_failed[s]
        used = [r for r in rows if r not in cold and r not in failed]
        lines.append(f"[{s}]  n_total={len(rows)}  "
                     f"used={len(used)}  cold-cache={len(cold)}  "
                     f"silent-OOM={len(failed)}")
        terms_str = " + ".join(
            f"{round_sig(v,3)}{'' if k=='1' else (f'·L' if k=='L' else '·L²')}"
            for k, v in fits[s].items())
        lines.append(f"  fit (wall_s) : {terms_str}")
        per_gpu = scaling.get(s, {}).get("per_gpu", {}).get(args.gpu, {})
        if per_gpu:
            if "runtime_sec_per_seq" in per_gpu:
                rt = per_gpu["runtime_sec_per_seq"]
                lines.append(f"  current      : base={rt.get('base',0)} "
                             f"alpha_L={rt.get('alpha_L',0)} "
                             f"alpha_L2={rt.get('alpha_L2',0)}")
            elif "runtime_sec_per_chunk" in per_gpu:
                rt = per_gpu["runtime_sec_per_chunk"]
                lines.append(f"  current      : base={rt.get('base',0)} "
                             f"per_seq={rt.get('per_seq',0)} "
                             f"per_residue={rt.get('per_residue',0)}")
        if used:
            residuals = [r["wall_s"] - predict(fits[s], [r["L"]])[0] for r in used]
            mae = float(np.mean(np.abs(residuals)))
            lines.append(f"  fit MAE      : {mae:.1f} s  ({len(used)} pts)")
        if failed:
            lines.append(f"  OOM seqs     : "
                         + ", ".join(f"{r['seq_id']}({int(r['L'])}aa)"
                                     for r in failed))
        lines.append("")

    summary_path = args.output_dir / "per_stage_summary.txt"
    summary_path.write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nWrote:")
    for p in (runtime_png, mem_png, chunk_png, out_yaml, chunk_csv, summary_path):
        print(f"  {p}")


if __name__ == "__main__":
    main()
