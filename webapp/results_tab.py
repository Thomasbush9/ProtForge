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

import streamlit as st

from session import Session
from pipeline_ops import load_config
from results import (
    list_result_sequences,
    find_structures,
    read_structure_text,
    structure_format,
    read_confidence,
    read_benchmarks,
)


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
    st.components.v1.html(view._make_html(), height=540)
    return True


def _structure_viewer(output_dir: str):
    seqs = list_result_sequences(output_dir)
    if not seqs:
        st.info(
            "No predicted structures found yet under "
            f"`{output_dir}/sequences/`. Structures appear here once Boltz / "
            "OpenFold3 / ESMFold2 finish."
        )
        return

    c1, c2 = st.columns([2, 2])
    with c1:
        seq = st.selectbox(f"Sequence ({len(seqs)} with results)", options=seqs,
                           key="results_seq")
    structures = find_structures(f"{output_dir}/sequences/{seq}")
    if not structures:
        st.warning("No structure files for this sequence.")
        return

    with c2:
        labels = [s.label for s in structures]
        idx = st.selectbox("Model", options=range(len(labels)),
                           format_func=lambda i: labels[i], key="results_model")
    chosen = structures[idx]

    color_mode = st.radio(
        "Colour by", ["pLDDT (b-factor)", "Rainbow (N→C)", "Chain"],
        horizontal=True, key="results_color",
    )

    # Confidence metrics for this model
    metrics = read_confidence(chosen)
    if metrics:
        # Surface the most informative scores first, then the rest.
        priority = ["confidence_score", "ranking_score", "ptm", "iptm",
                    "complex_plddt", "mean_plddt", "min_plddt"]
        ordered = [k for k in priority if k in metrics]
        ordered += [k for k in metrics if k not in ordered]
        cols = st.columns(min(len(ordered), 4) or 1)
        for i, k in enumerate(ordered[:8]):
            cols[i % len(cols)].metric(k, f"{metrics[k]:.3f}")

    try:
        text = read_structure_text(chosen.path)
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
    benches = read_benchmarks(output_dir)
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


def render_results_tab(session: Session):
    cfg = load_config(session)
    output_dir = cfg.get("output", {}).get("parent_dir", "")
    if not output_dir:
        st.info("Set an output directory in the Configuration tab to see results.")
        return

    from pathlib import Path
    if not Path(output_dir).is_dir():
        st.info(f"Output directory does not exist yet: `{output_dir}`")
        return

    view = st.radio("View", ["Structure viewer", "Run analytics"],
                    horizontal=True, key="results_view")
    st.divider()
    if view == "Structure viewer":
        _structure_viewer(output_dir)
    else:
        _run_analytics(output_dir)
