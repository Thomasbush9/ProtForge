#!/usr/bin/env python3
"""Fit the per-stage cost model that the chunk optimizer plans against.

The model, per stage:

    wall_s(job)  = overhead + scale · Σ_i f(L_i)
                 = overhead + scale · (a·N + b·ΣL + c·ΣL²)
    vram_gb(job) = v0 + v1·maxL + v2·maxL²
    host_gb(job) = r0 + r1·N + r2·ΣL

`overhead` is the whole point. It is the fixed per-job cost — container start,
model weights off Lustre, CUDA init — paid once no matter how much work the job
does. It is what makes "how many chunks?" a real optimization rather than a free
choice, and it is why a 70-second ESM-C job is almost entirely waste.

Two sources, used for what each is actually good for:
  * stair results.csv — single-sequence sweeps over L, with `infer_s` timed
    separately from wall. Gives the SHAPE of the per-sequence cost curve f(L)
    over a length range production never covers, and an independent read on
    overhead via (wall_total_s - infer_s).
  * observations.csv  — real production jobs. Gives the LEVEL: how that shape
    scales once sequences are batched, plus overhead, host RAM and real VRAM.

Fitting the shape from stair and only the level from production keeps the number
of free parameters down to two (overhead, scale), which matters because
production runs vary N over as few as two distinct values. Asking for more
parameters than the design supports is how you get a 75-minute "overhead" for a
28-minute job, so this module counts distinct design points and refuses to fit
a basis it cannot identify.

Coefficients are non-negative by construction: a negative overhead or a negative
per-residue cost is unphysical and would wreck the optimizer.

Usage:
    python -m scripts.calibrate.optimize.model --obs observations.csv \
        [--stair calib_h100_2k/results.csv ...] --out cost_model.yaml
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

# Stair stages whose infer_s IS wall time (no inner timer), so wall - infer
# carries no overhead information.
WALL_IS_INFER = {"msa", "boltz"}

# A job whose GPU sat this far below its stage's own median did not do the work
# it was given (model loaded, nothing folded). Its wall time is a pure-overhead
# observation, not a workload observation.
IDLE_ABS_PCT = 2.0
STAGE_ACTIVE_PCT = 30.0


def _f(v, default=0.0):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def nnls(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares (projected gradient; scipy-free).

    Columns are normalised before solving. The natural basis here is
    [1, L, L²], whose columns span ~1 to ~10⁶; on that scaling a projected
    gradient crawls along the small directions and returns coefficients that do
    not fit even the points it kept. Normalising makes the problem well
    conditioned, and the solution is mapped back exactly (the substitution
    z = scale·x preserves non-negativity because scale > 0).
    """
    if A.size == 0 or len(y) == 0:
        return np.zeros(A.shape[1] if A.ndim == 2 else 0)

    scale = np.linalg.norm(A, axis=0)
    scale[scale <= 0] = 1.0
    As = A / scale

    z, *_ = np.linalg.lstsq(As, y, rcond=None)
    if (z >= -1e-9).all():
        return np.maximum(z, 0.0) / scale

    z = np.maximum(z, 0.0)
    AtA, Aty = As.T @ As, As.T @ y
    step = 1.0 / (np.linalg.norm(AtA, 2) or 1.0)
    for _ in range(20000):
        z_new = np.maximum(z - step * (AtA @ z - Aty), 0.0)
        if np.allclose(z_new, z, rtol=1e-12, atol=1e-14):
            z = z_new
            break
        z = z_new
    return z / scale


def robust_nnls(A: np.ndarray, y: np.ndarray, min_keep: int = 4,
                n_sigma: float = 3.0, rounds: int = 3):
    """NNLS with MAD-based outlier rejection.

    A stalled or degraded run produces jobs whose wall time saturates at some
    ceiling regardless of workload. Those points are not noise around the true
    relationship — they come from a different process entirely, and left in they
    dominate the fit. Rejecting residuals beyond n_sigma robust deviations drops
    them without anyone having to hand-pick which run was bad.

    Returns (coef, keep_mask, n_dropped).
    """
    keep = np.ones(len(y), dtype=bool)
    coef = nnls(A, y)
    dropped = 0
    for _ in range(rounds):
        resid = y - A @ coef
        r = resid[keep]
        if len(r) < min_keep:
            break
        med = np.median(r)
        mad = float(np.median(np.abs(r - med))) * 1.4826
        if mad <= 1e-9:
            break
        new_keep = np.abs(resid - med) <= n_sigma * mad
        if new_keep.sum() < min_keep or (new_keep == keep).all():
            break
        keep = new_keep
        coef = nnls(A[keep], y[keep])
        dropped = int((~keep).sum())
    return coef, keep, dropped


def quality(A, y, coef) -> dict:
    if len(y) == 0:
        return {}
    resid = y - A @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"n": int(len(y)),
            "mae": round(float(np.abs(resid).mean()), 2),
            "r2": round(1 - ss_res / ss_tot, 3) if ss_tot > 1e-12 else None}


def _design_points(*cols) -> int:
    """Distinct rows across the given feature columns (rounded, so near-identical
    workloads do not masquerade as independent evidence)."""
    if not cols or len(cols[0]) == 0:
        return 0
    pts = {tuple(round(float(c[i]), 3) for c in cols) for i in range(len(cols[0]))}
    return len(pts)


# --- loading ---------------------------------------------------------------

def load_observations(path: Path) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            N, wall = _f(r.get("n_seqs")), _f(r.get("wall_s"))
            if N <= 0 or wall <= 0:
                continue
            gu = r.get("gpu_util_pct")
            by.setdefault(r["stage"], []).append({
                "N": N,
                "sumL": _f(r.get("total_residues")),
                "sumL2": _f(r.get("sum_len2")),
                "maxL": _f(r.get("max_len")),
                "wall": wall,
                "vram": _f(r.get("vram_gb"), None) if r.get("vram_gb") else None,
                "host": _f(r.get("host_gb")),
                "gpu_util": _f(gu, None) if gu not in (None, "") else None,
                "src": "prod",
            })
    return by


def load_stair(paths: list[Path]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    for p in paths:
        if not p.exists():
            continue
        with p.open() as f:
            for r in csv.DictReader(f):
                if r.get("status") != "ok":
                    continue
                stage = (r.get("stage") or "").split("_")[0]
                L = _f(r.get("seq_len"))
                if L <= 0:
                    continue
                vram_mib = _f(r.get("vram_peak_mib"), None)
                by.setdefault(stage, []).append({
                    "N": 1.0, "sumL": L, "sumL2": L * L, "maxL": L,
                    "wall": _f(r.get("wall_total_s"), None),
                    "infer": _f(r.get("infer_s"), None),
                    "vram": vram_mib / 1024 if vram_mib else None,
                    "host": None, "gpu_util": None, "src": "stair",
                })
    return by


# --- fitting ---------------------------------------------------------------

def fit_shape(stair: list[dict], stage: str) -> tuple[dict | None, dict]:
    """Per-sequence cost shape f(L) = a + b·L + c·L² from single-sequence rows."""
    pts = [r for r in stair if r.get("infer") is not None]
    if stage in WALL_IS_INFER:
        pts = [r for r in stair if r.get("wall") is not None]
        for r in pts:
            r.setdefault("infer", r["wall"])
    if len(pts) < 3:
        return None, {}
    L = np.array([r["maxL"] for r in pts])
    y = np.array([r["infer"] for r in pts])
    terms = 3 if _design_points(L) >= 4 else 2
    cols = [np.ones_like(L), L, L * L][:terms]
    A = np.vstack(cols).T
    c = nnls(A, y)
    coef = {"a": float(c[0]), "b": float(c[1]) if terms > 1 else 0.0,
            "c": float(c[2]) if terms > 2 else 0.0}
    return coef, quality(A, y, c)


def split_idle(prod: list[dict]) -> tuple[list[dict], list[dict], float | None]:
    """Separate jobs that plainly never did their work from the rest.

    A job sitting at ~0% GPU in a stage that normally runs hot loaded its model
    and then did nothing. Its wall time is a clean read on overhead, but its
    VRAM and runtime say nothing about the workload it was handed.
    """
    utils = [r["gpu_util"] for r in prod if r.get("gpu_util") is not None]
    if not utils:
        return [], list(prod), None
    med = float(np.median(utils))
    if med < STAGE_ACTIVE_PCT:
        return [], list(prod), med
    idle, working = [], []
    for r in prod:
        u = r.get("gpu_util")
        (idle if (u is not None and u < IDLE_ABS_PCT) else working).append(r)
    return idle, working, med


def fit_time(prod: list[dict], shape: dict | None, stage: str,
             idle: list[dict], working: list[dict], med_util: float | None,
             stair_overhead: float | None = None
             ) -> tuple[dict, dict, list[str]]:
    """Fit wall = overhead + scale·Σf(L), or the simplest identifiable fallback.

    `overhead` and the per-sequence term are only separable if N actually varies
    across jobs. A run that used one chunk size for every job cannot distinguish
    "300 s of startup" from "6 s per sequence × 50 sequences" — the regression
    will happily put it all in one term. When N is constant we therefore refuse
    the regression's intercept and take overhead from direct evidence instead:
    jobs that did no work (their whole wall IS overhead) or the stair sweep's
    wall-minus-inference gap. Getting this wrong would flip the optimizer's
    chunking advice, so it is worth the special case.
    """
    notes: list[str] = []
    if not prod:
        return {}, {}, notes

    direct_overhead = None
    if idle:
        direct_overhead = float(np.median([r["wall"] for r in idle]))
    elif stair_overhead:
        direct_overhead = float(stair_overhead)
    if idle:
        notes.append(
            f"{len(idle)} job(s) ran at <{IDLE_ABS_PCT}% GPU while this stage "
            f"normally runs at ~{med_util:.0f}% — treated as pure-overhead "
            f"observations, not workload observations")
    if not working:
        working = list(prod)

    N = np.array([r["N"] for r in working])
    sL = np.array([r["sumL"] for r in working])
    sL2 = np.array([r["sumL2"] for r in working])
    w = np.array([r["wall"] for r in working])

    if shape and len(working) >= 2:
        # Σf(L_i) = a·N + b·ΣL + c·ΣL² — computable from the stored features.
        work = shape["a"] * N + shape["b"] * sL + shape["c"] * sL2
        dp = _design_points(work)
        if dp >= 2:
            n_identifiable = _design_points(N) >= 2
            if not n_identifiable and direct_overhead is not None:
                # Fix overhead from direct evidence, fit only the scale.
                A = work.reshape(-1, 1)
                c, keep, dropped = robust_nnls(
                    A, np.maximum(w - direct_overhead, 0.0), min_keep=3)
                block = {"overhead_s": direct_overhead, "scale": float(c[0]),
                         "a": shape["a"], "b": shape["b"], "c": shape["c"]}
                q = quality(A[keep], (w - direct_overhead)[keep], c)
                q["method"] = ("stair shape + production level; overhead FIXED "
                               "from direct evidence (N constant, not identifiable)")
                notes.append(
                    f"every job used the same chunk size (N={int(N[0])}), so "
                    f"overhead is not identifiable from these runs — fixed at "
                    f"{direct_overhead:.0f}s from direct measurement")
            else:
                A = np.vstack([np.ones_like(work), work]).T
                c, keep, dropped = robust_nnls(A, w)
                block = {"overhead_s": float(c[0]), "scale": float(c[1]),
                         "a": shape["a"], "b": shape["b"], "c": shape["c"]}
                q = quality(A[keep], w[keep], c)
                q["method"] = "stair shape + production level (overhead, scale)"
            if dropped:
                q["dropped_outliers"] = dropped
                notes.append(
                    f"{dropped} job(s) rejected as outliers — wall time "
                    f"inconsistent with workload (degraded or stalled run)")
            return block, q, notes

    # No usable shape: fit the simplest basis the design can identify.
    dp_N = _design_points(N)
    if dp_N >= 3 and _design_points(N, sL2) >= 3:
        A = np.vstack([np.ones_like(N), N, sL2]).T
        c, keep, dropped = robust_nnls(A, w)
        block = {"overhead_s": float(c[0]), "scale": 1.0,
                 "a": float(c[1]), "b": 0.0, "c": float(c[2])}
        q = quality(A[keep], w[keep], c)
        q["method"] = "production: overhead + per-seq + per-L²"
        if dropped:
            q["dropped_outliers"] = dropped
    elif dp_N >= 2:
        A = np.vstack([np.ones_like(N), N]).T
        c = nnls(A, w)
        block = {"overhead_s": float(c[0]), "scale": 1.0,
                 "a": float(c[1]), "b": 0.0, "c": 0.0}
        q = quality(A, w, c)
        q["method"] = "production: overhead + per-seq (L did not vary)"
        notes.append(
            "sequence length was ~constant across these jobs, so cost-vs-length "
            "is UNCONSTRAINED here; run a stair sweep to pin it down")
    else:
        # One design point: everything we can say is "a job costs about this".
        block = {"overhead_s": float(np.median(w)), "scale": 1.0,
                 "a": 0.0, "b": 0.0, "c": 0.0}
        q = {"n": len(w), "method": "single design point — constant cost only"}
        notes.append(
            "only one distinct workload was ever run for this stage; the model "
            "is a constant and cannot extrapolate")

    # Idle jobs give a direct read on overhead; prefer it if the fit found none.
    if direct_overhead is not None and block["overhead_s"] <= 0:
        notes.append(
            f"regression put no cost in overhead; using direct measurement "
            f"({direct_overhead:.0f}s) instead")
        block["overhead_s"] = direct_overhead
    return block, q, notes


def fit_vram(rows: list[dict], idle_ids: set[int], prefer_prod: bool = True
             ) -> tuple[dict, dict]:
    """VRAM vs the longest sequence in the chunk.

    Production rows are preferred: they are batched, and a batched job's peak is
    not the same as a single sequence's. Jobs that never did their work are
    excluded — their "peak" is just the loaded weights and would flatten the
    curve badly.
    """
    prod = [r for r in rows
            if r["src"] == "prod" and r.get("vram") and id(r) not in idle_ids]
    stair = [r for r in rows if r["src"] == "stair" and r.get("vram")]
    use_prod = prefer_prod and len(prod) >= 3
    use = prod if use_prod else [r for r in (prod + stair) if r.get("vram")]
    if not use:
        return {}, {}
    L = np.array([r["maxL"] for r in use])
    y = np.array([r["vram"] for r in use])
    src = "production" if use_prod else "stair+production"
    if _design_points(L) < 3:
        return ({"base": round(float(max(y)), 3), "per_len": 0.0, "per_len2": 0.0},
                {"n": len(y), "method": f"max observed ({src}, too few lengths)"})
    terms = 3 if _design_points(L) >= 5 else 2
    cols = [np.ones_like(L), L, L * L][:terms]
    A = np.vstack(cols).T
    c, keep, dropped = robust_nnls(A, y)
    q = quality(A[keep], y[keep], c)
    q["method"] = f"{src}, {terms} terms"
    if dropped:
        q["dropped_outliers"] = dropped
    return ({"base": round(float(c[0]), 4),
             "per_len": round(float(c[1]) if terms > 1 else 0.0, 8),
             "per_len2": round(float(c[2]) if terms > 2 else 0.0, 12)}, q)


def fit_host(prod: list[dict]) -> tuple[dict, dict]:
    hr = [r for r in prod if r.get("host")]
    if not hr:
        return {}, {}
    N = np.array([r["N"] for r in hr])
    sL = np.array([r["sumL"] for r in hr])
    y = np.array([r["host"] for r in hr])
    if _design_points(N, sL) < 3:
        return ({"base": round(float(max(y)), 2), "per_seq": 0.0, "per_residue": 0.0},
                {"n": len(y), "method": "max observed (too few design points)"})
    A = np.vstack([np.ones_like(N), N]).T
    c = nnls(A, y)
    q = quality(A, y, c)
    q["method"] = "base + per-seq"
    return ({"base": round(float(c[0]), 3), "per_seq": round(float(c[1]), 6),
             "per_residue": 0.0}, q)


def fit_stage(stage: str, rows: list[dict]) -> dict:
    prod = [r for r in rows if r["src"] == "prod"]
    stair = [r for r in rows if r["src"] == "stair"]
    out: dict = {"n_obs": len(rows), "n_prod": len(prod), "n_stair": len(stair),
                 "sources": sorted({r["src"] for r in rows})}
    notes: list[str] = []

    shape, shape_q = fit_shape(stair, stage)
    if shape:
        out["per_seq_shape"] = {k: round(v, 8) for k, v in shape.items()}
        out["per_seq_shape_fit"] = shape_q

    # Stair's wall-minus-inference gap is an independent read on overhead.
    stair_overhead = None
    if stair and stage not in WALL_IS_INFER:
        gaps = [r["wall"] - r["infer"] for r in stair
                if r.get("wall") and r.get("infer")]
        if gaps:
            stair_overhead = float(np.median(gaps))

    idle, working, med_util = split_idle(prod)
    idle_ids = {id(r) for r in idle}
    time_block, time_q, tnotes = fit_time(prod, shape, stage, idle, working,
                                          med_util, stair_overhead)
    notes += tnotes
    if not time_block and shape and stair:
        gaps = [r["wall"] - r["infer"] for r in stair
                if r.get("wall") and r.get("infer") and stage not in WALL_IS_INFER]
        oh = max(float(np.median(gaps)), 0.0) if gaps else 0.0
        time_block = {"overhead_s": oh, "scale": 1.0, **shape}
        time_q = {"method": "stair wall-minus-infer (no production jobs)",
                  "n": len(gaps)}
    if time_block:
        out["time"] = {k: round(v, 8) for k, v in time_block.items()}
        out["time_fit"] = time_q

    if stair_overhead is not None:
        out["overhead_stair_check_s"] = round(stair_overhead, 1)
    if idle:
        out["overhead_idle_check_s"] = round(
            float(np.median([r["wall"] for r in idle])), 1)

    vram, vq = fit_vram(rows, idle_ids)
    if vram:
        out["vram_gb"], out["vram_fit"] = vram, vq
    host, hq = fit_host(working or prod)
    if host:
        out["host_gb"], out["host_fit"] = host, hq

    gu = [r["gpu_util"] for r in prod if r.get("gpu_util") is not None]
    if gu:
        out["gpu_util_pct"] = {"median": round(float(np.median(gu)), 1),
                               "max": round(float(max(gu)), 1)}

    # The length range the fit actually saw. Predictions outside it are
    # extrapolation, and the optimizer says so rather than quoting a number
    # with false confidence.
    seen = [r["maxL"] for r in rows if r.get("maxL")]
    if seen:
        out["fitted_len_range"] = [int(min(seen)), int(max(seen))]
    seen_n = [r["N"] for r in prod if r.get("N")]
    if seen_n:
        out["fitted_n_range"] = [int(min(seen_n)), int(max(seen_n))]

    if notes:
        out["notes"] = notes
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", type=Path, required=True)
    ap.add_argument("--stair", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, default=Path("cost_model.yaml"))
    args = ap.parse_args()

    by = load_observations(args.obs)
    for stage, rows in load_stair(args.stair).items():
        by.setdefault(stage, []).extend(rows)
    if not by:
        print("no usable observations", file=sys.stderr)
        return 1

    model = {stage: fit_stage(stage, rows) for stage, rows in sorted(by.items())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(model, sort_keys=False,
                                       default_flow_style=False))

    print(f"fitted {len(model)} stages -> {args.out}\n")
    for stage, m in model.items():
        print(f"[{stage}]  prod={m['n_prod']} stair={m['n_stair']}")
        t = m.get("time")
        if t:
            oh, sc = t.get("overhead_s", 0), t.get("scale", 1.0)
            print(f"   overhead : {oh:8.1f} s/job  ({oh/60:5.1f} min paid per job)")
            chk = m.get("overhead_stair_check_s")
            if chk is not None:
                print(f"              (stair cross-check: {chk:.0f} s)")
            print(f"   per-seq  : {sc:.3f}·[{t.get('a',0):.3g} + {t.get('b',0):.3g}·L "
                  f"+ {t.get('c',0):.3g}·L²] s")
            q = m.get("time_fit", {})
            print(f"   time fit : MAE {q.get('mae','?')}s  R²={q.get('r2')}  "
                  f"[{q.get('method','')}]")
        if "vram_gb" in m:
            v, q = m["vram_gb"], m.get("vram_fit", {})
            print(f"   VRAM     : {v['base']:.1f} + {v['per_len']:.4g}·maxL "
                  f"+ {v['per_len2']:.2g}·maxL² GB  R²={q.get('r2')} [{q.get('method','')}]")
        if "host_gb" in m:
            h, q = m["host_gb"], m.get("host_fit", {})
            print(f"   host RAM : {h['base']:.1f} + {h['per_seq']:.4g}·N GB  "
                  f"R²={q.get('r2')} [{q.get('method','')}]")
        if "gpu_util_pct" in m:
            g = m["gpu_util_pct"]
            print(f"   GPU util : median {g['median']}%  max {g['max']}%")
        for n in m.get("notes", []):
            print(f"   ! {n}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
