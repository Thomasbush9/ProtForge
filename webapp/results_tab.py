"""
Results tab for the ProtForge web UI.

Two sections over the active session's output tree:
  - Structure viewer: browse completed sequences, render a predicted CIF/PDB
    in 3D (py3Dmol), coloured by pLDDT, with per-model confidence metrics.
  - Run analytics: per-stage runtime distributions, node-hours, and peak
    memory parsed from the Snakemake benchmark TSVs.

py3Dmol and plotly are optional; the tab degrades to a download link / native
charts and prints an install hint when they're missing.
"""

from pathlib import Path

import streamlit as st

from session import Session
from pipeline_ops import load_config
from results import (
    StructureFile,
    list_sequence_dirs,
    find_structures,
    read_structure_text,
    structure_format,
    model_summary,
    read_benchmarks,
)
from structure_compare import compare

# How many sequences to put in the picker before asking the user to narrow the
# filter — keeps the selectbox responsive on runs with thousands of mutants.
_MAX_PICKER = 300

# Filesystem reads are cached with a short TTL so reruns (widget changes,
# autorefresh) don't re-walk the output tree on the network filesystem. The
# Refresh button clears them when the user wants fresh state.
_RESULTS_TTL = 30


@st.cache_data(ttl=_RESULTS_TTL, show_spinner=False)
def _seq_dirs(output_dir: str) -> list[str]:
    return list_sequence_dirs(output_dir)


@st.cache_data(ttl=_RESULTS_TTL, show_spinner=False)
def _structures(output_dir: str, seq: str) -> list[StructureFile]:
    return find_structures(Path(output_dir) / "sequences" / seq)


@st.cache_data(ttl=_RESULTS_TTL, show_spinner=False)
def _summary(path_str: str, stage: str) -> dict:
    return model_summary(StructureFile(stage=stage, path=Path(path_str), label=""))


@st.cache_data(show_spinner=False)
def _structure_text(path_str: str, _mtime: float) -> str:
    # _mtime is part of the cache key so an overwritten file invalidates.
    return read_structure_text(path_str)


@st.cache_data(ttl=_RESULTS_TTL, show_spinner=False)
def _benchmarks(output_dir: str) -> dict:
    return read_benchmarks(output_dir)


def _clear_results_cache() -> None:
    for fn in (_seq_dirs, _structures, _summary, _structure_text, _benchmarks):
        fn.clear()


# pLDDT colour scheme (AlphaFold-style): blue = confident, orange/red = low.
_PLDDT_COLORSCHEME = {
    "prop": "b",  # b-factor column carries pLDDT for these predictors
    "gradient": "roygb",
    "min": 50,
    "max": 90,
}


def _render_3d(structure_text: str, fmt: str, color_mode: str) -> bool:
    """Embed a py3Dmol viewer. Returns False if py3Dmol isn't installed."""
    try:
        import py3Dmol
    except ImportError:
        return False

    view = py3Dmol.view(width=720, height=520)
    view.addModel(structure_text, fmt)
    if color_mode == "pLDDT (b-factor)":
        view.setStyle({"cartoon": {"colorscheme": _PLDDT_COLORSCHEME}})
    elif color_mode == "Chain":
        view.setStyle({"cartoon": {"colorscheme": "chain"}})
    else:  # Rainbow (N→C)
        view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    view.setBackgroundColor("0xffffff")
    # st.iframe auto-detects the raw HTML string and embeds it in an iframe
    # (replaces the deprecated st.components.v1.html; keeps JS execution).
    st.iframe(view._make_html(), height=540)
    return True


def _structure_viewer(output_dir: str):
    all_dirs = _seq_dirs(output_dir)
    if not all_dirs:
        st.info(
            f"No sequences under `{output_dir}/sequences/` yet. Structures "
            "appear here once Boltz / OpenFold3 / ESMFold2 finish."
        )
        return

    c1, c2 = st.columns([2, 2])
    with c1:
        query = st.text_input(
            "Filter sequences", value="", key="results_seq_filter",
            placeholder="substring match…",
            help="Narrow the picker by name. Structures are resolved only for "
                 "the selected sequence, so this stays fast on large runs.",
        )
    filtered = [d for d in all_dirs if query.lower() in d.lower()] if query else all_dirs
    if not filtered:
        st.warning(f"No sequence names match `{query}`.")
        return
    shown = filtered[:_MAX_PICKER]
    with c2:
        seq = st.selectbox(
            f"Sequence ({len(filtered)} of {len(all_dirs)} shown)"
            if query else f"Sequence ({len(all_dirs)} total)",
            options=shown, key="results_seq",
        )
    if len(filtered) > _MAX_PICKER:
        st.caption(
            f"Showing first {_MAX_PICKER} of {len(filtered)} matches — "
            "type more to narrow."
        )

    structures = _structures(output_dir, seq)
    if not structures:
        st.warning(
            f"No structure files for `{seq}` yet — its predictions may still be "
            "running, or only embeddings (ESMC) were produced."
        )
        return

    with c2:
        labels = [s.label for s in structures]
        idx = st.selectbox("Model", options=range(len(labels)),
                           format_func=lambda i: labels[i], key="results_model")
    chosen = structures[idx]

    # Normalized summary (uniform across Boltz / OpenFold3 / ESMFold2).
    summary = _summary(str(chosen.path), chosen.stage)

    # Headline confidence: same fields regardless of predictor.
    headline = [(lbl, summary.get(key)) for lbl, key in (
        ("pLDDT", "plddt_mean"), ("pTM", "ptm"),
        ("ipTM", "iptm"), ("Ranking", "ranking_score"))]
    headline = [(lbl, v) for lbl, v in headline if v is not None]
    if headline:
        cols = st.columns(len(headline))
        for i, (lbl, v) in enumerate(headline):
            cols[i].metric(lbl, f"{v:.2f}")
    raw = summary.get("metrics") or {}
    if raw:
        with st.expander("All raw metrics"):
            st.json(raw)

    color_mode = st.radio(
        "Colour by", ["pLDDT (b-factor)", "Rainbow (N→C)", "Chain"],
        horizontal=True, key="results_color",
    )
    if color_mode.startswith("pLDDT"):
        plddt_ok = summary.get("plddt_in_bfactor")
        if plddt_ok is False:
            st.warning(
                "This model's B-factor column doesn't look like pLDDT "
                "(constant or out of 0-100) — the colouring may be meaningless. "
                "Try Rainbow / Chain."
            )
        elif plddt_ok is None:
            st.caption("Note: pLDDT-in-B-factor couldn't be verified for this file.")

    try:
        mtime = chosen.path.stat().st_mtime
        text = _structure_text(str(chosen.path), mtime)
    except Exception as exc:
        st.error(f"Could not read structure: {exc}")
        return

    rendered = _render_3d(text, structure_format(chosen.path), color_mode)
    if not rendered:
        st.warning(
            "3D viewer needs **py3Dmol** — install it on the host env:\n"
            "`pip install py3Dmol` (or `pip install '.[viz]'`). "
            "You can still download the structure below."
        )
    st.download_button(
        "Download structure", data=text, file_name=chosen.path.name,
        mime="chemical/x-cif", key="results_download",
    )
    st.caption(f"`{chosen.path}`")


def _run_analytics(output_dir: str):
    benches = _benchmarks(output_dir)
    if not benches:
        st.info(
            f"No benchmark data under `{output_dir}/benchmarks/`. "
            "Each completed rule writes a per-job TSV the pipeline aggregates here."
        )
        return

    total_node_h = sum(b.node_hours for b in benches.values())
    total_jobs = sum(b.n_jobs for b in benches.values())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total node-hours", f"{total_node_h:.1f}")
    c2.metric("Total jobs", total_jobs)
    c3.metric("Stages run", len(benches))

    rows = [{
        "Stage": b.stage.upper(),
        "Jobs": b.n_jobs,
        "Node-hours": round(b.node_hours, 2),
        "Mean (min)": round(b.mean_s / 60, 1),
        "p95 (min)": round(b.p95_s / 60, 1),
        "Max (min)": round(b.max_s / 60, 1),
        "Peak mem (GB)": round(b.max_rss_mb / 1024, 1),
    } for b in benches.values()]
    st.dataframe(rows, width="stretch", hide_index=True)

    # Per-stage runtime distribution. Plotly box plots if available; else a
    # native bar chart of node-hours per stage.
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for b in benches.values():
            fig.add_trace(go.Box(
                y=[s / 60 for s in b.runtimes_s], name=b.stage.upper(),
                boxpoints="outliers",
            ))
        fig.update_layout(
            yaxis_title="Per-job wall-clock (min)",
            showlegend=False, height=420, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.caption(
            "Install **plotly** (`pip install '.[viz]'`) for per-job runtime "
            "distributions. Showing node-hours per stage instead:"
        )
        st.bar_chart({b.stage.upper(): b.node_hours for b in benches.values()})


def _structure_comparison(output_dir: str):
    all_dirs = _seq_dirs(output_dir)
    if not all_dirs:
        st.info(f"No sequences under `{output_dir}/sequences/` yet.")
        return

    st.caption(
        "Pick a **target** structure and one or more **queries**, then compute "
        "RMSD (same-length, residue-by-residue) and TM-score (alignment-based). "
        "Defaults to the target sequence's other models — i.e. cross-predictor "
        "agreement; add other sequences to compare designs (e.g. mutant vs WT)."
    )

    # --- Target structure ---
    c1, c2 = st.columns(2)
    with c1:
        tq = st.text_input("Filter target sequence", value="", key="cmp_tfilter",
                           placeholder="substring…")
        tdirs = [d for d in all_dirs if tq.lower() in d.lower()] if tq else all_dirs
        if not tdirs:
            st.warning(f"No sequence matches `{tq}`.")
            return
        tseq = st.selectbox("Target sequence", tdirs[:_MAX_PICKER], key="cmp_tseq")
    target_structs = _structures(output_dir, tseq)
    if not target_structs:
        st.warning(f"No structures for `{tseq}`.")
        return
    with c2:
        tlabels = [s.label for s in target_structs]
        tidx = st.selectbox("Target model", range(len(tlabels)),
                            format_func=lambda i: tlabels[i], key="cmp_tmodel")
    target = target_structs[tidx]

    # --- Query structures (across one or more sequences) ---
    qfilter = st.text_input("Filter query sequences", value="", key="cmp_qfilter",
                            placeholder="substring… (leave blank to use the target sequence)")
    qpool = [d for d in all_dirs if qfilter.lower() in d.lower()] if qfilter else all_dirs
    qseqs = st.multiselect("Query sequence(s)", qpool[:_MAX_PICKER],
                           default=[tseq] if tseq in qpool else [], key="cmp_qseqs")

    query_items = []  # (label, StructureFile)
    for qs in qseqs:
        for s in _structures(output_dir, qs):
            if qs == tseq and s.path == target.path:
                continue  # don't compare the target to itself
            query_items.append((f"{qs} / {s.label}", s))
    if not query_items:
        st.info("Select at least one query sequence (its structures become queries).")
        return

    labels = [lab for lab, _ in query_items]
    chosen = st.multiselect("Queries", labels, default=labels[:10], key="cmp_queries")
    if not chosen:
        st.info("Pick one or more queries.")
        return

    if st.button("Compute comparison", type="primary", key="cmp_go"):
        tgt_text = _structure_text(str(target.path), target.path.stat().st_mtime)
        rows, tm_missing = [], False
        for lab, s in query_items:
            if lab not in chosen:
                continue
            q_text = _structure_text(str(s.path), s.path.stat().st_mtime)
            r = compare(tgt_text, q_text)
            tm_missing = tm_missing or any("tmtools" in n for n in r["notes"])
            rows.append({
                "Query": lab,
                "Target len": r["target_len"],
                "Query len": r["query_len"],
                "RMSD (Å)": round(r["rmsd"], 3) if r["rmsd"] is not None else "—",
                "TM (norm target)": round(r["tm_norm_target"], 3) if r["tm_norm_target"] is not None else "—",
                "TM (norm query)": round(r["tm_norm_query"], 3) if r["tm_norm_query"] is not None else "—",
            })
        st.caption(f"Target: `{target.label}` ({tseq})")
        st.dataframe(rows, width="stretch", hide_index=True)
        if tm_missing:
            st.caption(
                "TM-score shown as — because **tmtools** isn't installed "
                "(`pip install tmtools` or `pip install '.[viz]'`). RMSD needs "
                "no extra dependency."
            )


def render_results_tab(session: Session):
    cfg = load_config(session)
    output_dir = cfg.get("output", {}).get("parent_dir", "")
    if not output_dir:
        st.info("Set an output directory in the Configuration tab to see results.")
        return

    if not Path(output_dir).is_dir():
        st.info(f"Output directory does not exist yet: `{output_dir}`")
        return

    c_view, c_refresh = st.columns([4, 1])
    with c_view:
        view = st.radio("View", ["Structure viewer", "Compare structures", "Run analytics"],
                        horizontal=True, key="results_view")
    with c_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↻ Refresh", width="stretch", key="results_refresh",
                     help=f"Results are cached for ~{_RESULTS_TTL}s; click to "
                          "re-read the output tree now."):
            _clear_results_cache()
            st.rerun()
    st.divider()
    if view == "Structure viewer":
        _structure_viewer(output_dir)
    elif view == "Compare structures":
        _structure_comparison(output_dir)
    else:
        _run_analytics(output_dir)
