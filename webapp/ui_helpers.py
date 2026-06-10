"""
Reusable Streamlit widgets for the ProtForge Configuration tab.

Per-stage controls (GPU preference, SLURM resources, concurrency caps,
bin-aware chunking) and the resource-estimate panel. Each function reads/writes
the passed-in config dict; persistence is the caller's job (Save button) except
where noted (the estimate panel's "Apply" writes the config directly).
"""

from pathlib import Path

import streamlit as st

from validate import scan_directory
from session import Session, touch_session
from estimator import (
    apply_estimate_to_config,
    compute_input_stats,
    estimate_all_stages,
    load_scaling_models,
)


@st.cache_data(show_spinner="Scanning directory…")
def _cached_scan(path_str: str, dir_mtime: float) -> dict:
    """scan_directory wrapped with a cache key that invalidates when the dir
    is touched. mtime alone is good enough for the common case (adding or
    removing files); the explicit Re-scan button below covers content edits."""
    return scan_directory(Path(path_str))


def autoscan_directory(path_str: str) -> dict | None:
    """Return a scan_directory() result for `path_str`, or None if it's not
    a usable directory. Cached so typing in the path field stays snappy."""
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_dir():
        return None
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    return _cached_scan(path_str, mtime)


def render_gpu_preference(stage: str) -> None:
    """Per-stage GPU dropdown that persists in st.session_state["gpu_preferences"]."""
    if stage not in {"msa", "boltz", "esmc", "esmfold", "openfold"}:
        return
    prefs = st.session_state.setdefault("gpu_preferences", {})
    options = ["auto", "a100", "h100"]
    current = prefs.get(stage, "auto")
    if current not in options:
        current = "auto"
    chosen = st.selectbox(
        "GPU preference",
        options=options,
        index=options.index(current),
        key=f"gpu_pref_{stage}",
        help="auto = pick the cheapest GPU whose memory ceiling covers the "
             "estimated need. Pin a card if you have a specific requirement.",
    )
    prefs[stage] = chosen


def render_chunk_recommendation(stage: str) -> None:
    """If a recent estimate is cached, show 'Recommended: N' for this stage."""
    est = st.session_state.get("last_estimate", {}).get(stage)
    if not est:
        return
    st.caption(
        f"Recommended files per job: **{est['chunk_size']}** "
        f"({est['num_chunks']} jobs, ~{est['runtime_min']} min/job)"
    )


# Per-stage SLURM resource defaults — kept in sync with the rule fallbacks so a
# blank session picks up reasonable values without forcing the user to run the
# estimator first. Tuples are (mem_mb, runtime_min, cpus_per_task).
_SLURM_DEFAULTS: dict[str, tuple[int, int, int]] = {
    "msa":      (256000,  60, 4),
    "boltz":    ( 16000,  60, 8),
    "esmc":     (128000, 120, 8),
    "esmc_sae": (128000, 120, 8),
    "esmfold":  (128000, 120, 8),
    "openfold": ( 48000,  60, 8),
}


def render_slurm_resources(cfg: dict, stage: str,
                           defaults: tuple[int, int, int] | None = None) -> None:
    """Render mem / runtime / cpus number_inputs for a stage.

    Reads/writes cfg['slurm']['resources'][stage]. The webapp estimator's
    'Apply to session config' button populates the same block; values typed
    here win on save, so this is also the manual-override surface. `defaults`
    overrides the fallback tuple — used for per-size keys like esmc_6B that
    aren't in _SLURM_DEFAULTS."""
    if defaults is None:
        if stage not in _SLURM_DEFAULTS:
            return
        defaults = _SLURM_DEFAULTS[stage]
    default_mem, default_runtime, default_cpus = defaults
    slurm = cfg.setdefault("slurm", {})
    resources = slurm.setdefault("resources", {})
    stage_res = resources.setdefault(stage, {})
    c1, c2, c3 = st.columns(3)
    stage_res["mem_mb"] = c1.number_input(
        "Memory (MB)",
        value=int(stage_res.get("mem_mb", default_mem)),
        min_value=1000,
        step=1000,
        key=f"{stage}_mem_mb_override",
        help="Per-job SLURM mem request. Estimator's 'Apply' button writes here; "
             "you can also override manually. Ensure the target partition can "
             "actually serve this size.",
    )
    stage_res["runtime"] = c2.number_input(
        "Runtime (min)",
        value=int(stage_res.get("runtime", default_runtime)),
        min_value=1,
        key=f"{stage}_runtime_override",
    )
    stage_res["cpus_per_task"] = c3.number_input(
        "CPUs per task",
        value=int(stage_res.get("cpus_per_task", default_cpus)),
        min_value=1,
        key=f"{stage}_cpus_override",
    )


def render_max_concurrent(stage_cfg: dict, stage: str) -> None:
    """Optional cap on how many of this stage's jobs run at once.

    Writes <stage>.max_concurrent_jobs; launch_snakemake turns it into
    `--resources <stage>_jobs=N`. Off = unbounded (up to the profile's global
    `jobs:` limit). `stage_cfg` is the per-stage dict the caller saves back."""
    on = st.toggle(
        "Limit concurrent jobs",
        value=stage_cfg.get("max_concurrent_jobs") is not None,
        help="Cap how many of this stage's SLURM jobs run simultaneously "
             "(Snakemake --resources). Useful to avoid flooding the scheduler "
             "or to bound GPU usage (e.g. OpenFold jobs each take several GPUs).",
        key=f"{stage}_cap_concurrency",
    )
    if on:
        stage_cfg["max_concurrent_jobs"] = int(st.number_input(
            "Max concurrent jobs",
            value=int(stage_cfg.get("max_concurrent_jobs") or 4), min_value=1,
            key=f"{stage}_max_concurrent_jobs"))
    else:
        stage_cfg.pop("max_concurrent_jobs", None)


def render_binning_controls(cfg: dict, stage: str) -> None:
    """Per-stage 'Bin-aware chunking' toggle + mode + preview table.

    The bins recipe (chunk_size/mem/runtime per bin) is populated by the
    estimator's 'Apply to session config' button. The UI toggles enabled +
    mode + num_bins; per-bin numbers are read-only here (edit config.yaml
    for fine-grained overrides). Only MSA/Boltz support binning — the ESMC /
    ESMFold2 chunkers split by max_files_per_job only.
    """
    if stage not in {"msa", "boltz"}:
        return
    stage_cfg = cfg.setdefault(stage, {})
    binning = stage_cfg.setdefault("binning", {})

    enabled = st.toggle(
        "Bin-aware chunking",
        value=bool(binning.get("enabled", False)),
        help="Partition sequences into length bins; each bin produces chunks "
             "with bin-specific SLURM mem and runtime. Recipe populated by "
             "the estimator (click 'Apply to session config' after enabling).",
        key=f"{stage}_binning_enabled",
    )
    binning["enabled"] = enabled
    if not enabled:
        return

    c1, c2, c3 = st.columns([1, 1, 1])
    mode = c1.selectbox(
        "Bin mode",
        options=["quantile", "thresholds"],
        index=0 if binning.get("mode", "quantile") == "quantile" else 1,
        help="quantile: cuts derived from your input length distribution. "
             "thresholds: explicit cuts (set in 'Length cuts' below).",
        key=f"{stage}_binning_mode",
    )
    binning["mode"] = mode
    if mode == "quantile":
        binning["num_bins"] = c2.number_input(
            "Number of bins",
            value=int(binning.get("num_bins", 6)),
            min_value=2,
            max_value=10,
            help="6 uses upper-tail-weighted cuts (q25/q50/q75/q90/q95). "
                 "Other values use evenly-spaced quantiles.",
            key=f"{stage}_binning_num_bins",
        )
    else:
        thresholds_str = ",".join(str(int(t)) for t in (binning.get("thresholds") or []))
        new_str = c2.text_input(
            "Length cuts",
            value=thresholds_str,
            help="Comma-separated. Example: 400,800,1200,1800 -> 5 bins.",
            key=f"{stage}_binning_thresholds",
        )
        try:
            binning["thresholds"] = [int(x.strip()) for x in new_str.split(",") if x.strip()]
            binning["num_bins"] = len(binning["thresholds"]) + 1
        except ValueError:
            st.warning("Thresholds must be integers separated by commas.")

    binning["chunks_per_bin"] = c3.number_input(
        "Chunks per bin",
        value=int(binning.get("chunks_per_bin", 1)),
        min_value=1,
        help="How many parallel chunks each non-empty bin is split into. "
             "Total parallel jobs ≈ num_bins × chunks_per_bin (capped at "
             "bin_count for sparse bins).",
        key=f"{stage}_binning_chunks_per_bin",
    )

    # Preview from the latest estimate (if any). last_estimate stores dict
    # form (asdict of StageEstimate), so bin_plan here is a dict or None.
    est = st.session_state.get("last_estimate", {}).get(stage)
    bin_plan = est.get("bin_plan") if isinstance(est, dict) else None
    if bin_plan and bin_plan.get("bins"):
        st.caption("Estimated plan (re-run estimator after changing input data):")
        rows = []
        total_chunks = 0
        total_runtime = 0
        cpb = bin_plan.get("chunks_per_bin", 1)
        for b in bin_plan["bins"]:
            n = b["num_seqs"]
            n_chunks = min(cpb, n) if n else 0
            cs = b.get("chunk_size") or (((n + n_chunks - 1) // n_chunks) if n_chunks else 0)
            total_chunks += n_chunks
            total_runtime += n_chunks * b["runtime_min"]
            rows.append({
                "bin": b["bin_idx"],
                "n": n,
                "L range": f"{b['len_lo']}-{b['len_hi']}",
                "chunks": n_chunks,
                "seqs/chunk": cs,
                "mem_mb": b["mem_mb"],
                "runtime_min": b["runtime_min"],
            })
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            f"Total chunks: {total_chunks} (≈ {bin_plan.get('num_bins', 0)} bins × "
            f"{cpb} chunks/bin), total runtime budget: {total_runtime} min "
            f"(thresholds: {bin_plan.get('thresholds', [])})"
        )
    else:
        st.caption("No bin plan yet — run the estimator to populate per-bin recipes.")


def render_estimate_panel(scan_result: dict, cfg: dict, session: Session,
                          key_prefix: str = "est") -> None:
    """Render the resource-estimate expander given a scan_directory() result.

    Computes input stats, calls the estimator for each enabled pipeline stage,
    shows a per-stage table, and offers an "Apply to session config" button.
    Caches stats + estimate in st.session_state for the Configuration tab.
    """
    file_type = scan_result.get("file_type")
    if file_type == "fasta":
        stats = compute_input_stats(fasta_results=scan_result.get("fasta_results", []))
    elif file_type == "yaml":
        stats = compute_input_stats(yaml_results=scan_result.get("yaml_results", []))
    else:
        return

    if stats.count == 0:
        return

    # GPU preferences from session_state (set by Configuration tab dropdowns).
    gpu_prefs = st.session_state.get("gpu_preferences", {})

    try:
        scaling = load_scaling_models()
        estimates = estimate_all_stages(stats, cfg, scaling, gpu_prefs)
    except Exception as exc:
        st.error(f"Resource estimate failed: {exc}")
        return

    # Cache for Configuration tab
    st.session_state["last_input_stats"] = stats.as_dict()
    st.session_state["last_estimate"] = {s: e.as_dict() for s, e in estimates.items()}

    if not estimates:
        st.info(
            f"Found {stats.count} valid {stats.file_type.upper()} file(s), "
            "but no pipeline stages are enabled. Toggle MSA / Boltz / ESMC / "
            "ESMFold2 above to see resource estimates."
        )
        return

    total_node_h = sum(e.total_node_hours for e in estimates.values())
    total_jobs = sum(e.num_chunks for e in estimates.values())

    with st.expander(
        f"Resource estimate — {total_node_h:.1f} node-hours across {total_jobs} jobs",
        expanded=True,
    ):
        st.caption(
            f"Based on {stats.count} sequence(s): "
            f"mean length {stats.mean_len:.0f}, p95 {stats.p95_len}, max {stats.max_len}."
        )

        rows = []
        any_notes = False
        for stage, e in estimates.items():
            rows.append({
                "Stage": stage.upper(),
                "Mem (GB)": round(e.mem_mb / 1024, 1),
                "Runtime (min)": e.runtime_min,
                "CPUs": e.cpus,
                "GPU": e.gpu_type or "-",
                "Partition": e.partition or "-",
                "Chunk size": e.chunk_size,
                "# Jobs": e.num_chunks,
                "Node-hours": e.total_node_hours,
            })
            if e.notes:
                any_notes = True
        st.dataframe(rows, width="stretch", hide_index=True)

        if any_notes:
            with st.expander("Notes from estimator", expanded=False):
                for stage, e in estimates.items():
                    for n in e.notes:
                        st.write(f"- **{stage}**: {n}")

        col_apply, col_info = st.columns([2, 3])
        with col_apply:
            if st.button(
                "Apply estimates to session config",
                key=f"{key_prefix}_apply",
                type="secondary",
                width="stretch",
            ):
                try:
                    apply_estimate_to_config(session.config_path, estimates, backup=True)
                    touch_session(session.id)
                    st.success(
                        "Wrote slurm.resources.<stage> + chunk sizes to "
                        f"`{session.config_path.name}` (backup saved)."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not apply estimates: {exc}")
        with col_info:
            st.caption(
                "Writes per-stage mem/runtime/cpus/gpus to slurm.resources, "
                "partition under slurm.<stage>.partition, and chunk size into "
                "the stage's own block."
            )
