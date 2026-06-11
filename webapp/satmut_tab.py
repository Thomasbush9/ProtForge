"""
Saturation Mutagenesis tab — a standalone single-sequence workflow.

Paste one sequence, pick an ESM-C size, launch a scan job, and inspect the
Len x 20-AA log-likelihood-ratio (LLR) matrix as an interactive heatmap with
CSV download. Independent of the pipeline session config (it only borrows the
cluster/container settings from it).
"""

import json
from pathlib import Path

import streamlit as st

from session import Session
from pipeline_ops import load_config
from validate import VALID_AAS
import satmut


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())


def _heatmap(out_dir: str, name: str, size: str):
    csv_path = satmut.matrix_csv(out_dir, name, size)

    # Derive the matrix from logits if the job finished but the CSV isn't built.
    if not csv_path.is_file() and satmut.logits_path(out_dir, name, size).is_file():
        seq = _query_seq(out_dir, name)
        if seq:
            try:
                satmut.derive_scan(out_dir, name, size, seq)
            except Exception as exc:
                st.error(f"Could not derive matrix from logits: {exc}")
                return

    if not csv_path.is_file():
        return False

    wt, aa_order, mat = satmut.load_matrix(csv_path)
    st.caption(f"`{csv_path}` — {len(wt)} positions × {len(aa_order)} substitutions")

    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Heatmap(
            z=mat.T,                                # (20 AAs, L positions)
            x=list(range(1, len(wt) + 1)),
            y=aa_order,
            colorscale="RdBu", zmid=0.0,
            colorbar=dict(title="LLR"),
            hovertemplate="pos %{x} (wt %{customdata})<br>→ %{y}<br>LLR %{z:.2f}<extra></extra>",
            customdata=[list(wt)] * len(aa_order),
        ))
        fig.update_layout(
            height=540, xaxis_title="residue position",
            yaxis_title="substitution", margin=dict(t=20, b=20),
        )
        # Interactive: zoom/pan + a PNG export button in the plotly modebar.
        st.plotly_chart(fig, use_container_width=True,
                        config={"toImageButtonOptions": {"filename": f"{name}_{size}_satmut"}})
    except ImportError:
        st.caption("Install **plotly** (`pip install '.[viz]'`) for the heatmap; "
                   "showing the raw matrix instead.")
        st.dataframe(mat, width="stretch")

    st.download_button(
        "Download matrix (CSV)", data=csv_path.read_bytes(),
        file_name=f"{name}_{size}_mutation_scan.csv", mime="text/csv",
        key="satmut_download",
    )
    return True


def _query_seq(out_dir: str, name: str) -> str | None:
    fa = satmut.input_fasta(out_dir, name)
    if not fa.is_file():
        return None
    lines = [l.strip() for l in fa.read_text().splitlines() if l.strip() and not l.startswith(">")]
    return "".join(lines) or None


def render_satmut_tab(session: Session):
    cfg = load_config(session)

    st.subheader("Saturation Mutagenesis (ESM-C)")
    st.caption(
        "Score every single substitution of one sequence with ESM-C: a "
        "wt-marginal log-likelihood ratio per position × amino acid "
        "(negative = less likely than wild-type). Runs one GPU forward pass, "
        "then renders the matrix here. Separate from the pipeline — it only "
        "uses the container/cache/SLURM settings from the active session."
    )

    # --- Input ---
    c1, c2 = st.columns([1, 2])
    with c1:
        name = st.text_input("Name", value="", key="satmut_name",
                             placeholder="e.g. gfp_wt")
    with c2:
        seq_raw = st.text_area("Sequence (one protein, amino acids only)",
                               height=120, key="satmut_seq")
    seq = "".join(seq_raw.split()).upper()

    # --- Config ---
    c1, c2 = st.columns([1, 2])
    with c1:
        size = st.selectbox("ESM-C model", satmut.ESMC_SIZES, index=0, key="satmut_size")
    default_out = ""
    parent = cfg.get("output", {}).get("parent_dir", "")
    if parent:
        default_out = str(Path(parent) / "satmut")
    with c2:
        out_dir = st.text_input("Output directory", value=default_out, key="satmut_out",
                               help="The matrix + logits are written under here, in <name>/<size>/.")

    # --- Validation ---
    safe = _safe_name(name)
    errors = []
    if seq:
        bad = sorted(set(seq) - VALID_AAS)
        if bad:
            errors.append(f"Invalid amino acids: {', '.join(bad)}")
    cluster = satmut.resolve_cluster_settings(cfg)
    can_launch = bool(seq and safe and out_dir and not errors and not cluster["errors"])

    if seq and not errors:
        st.caption(f"{len(seq)} residues · scan will produce a {len(seq)} × 20 matrix")
    for e in errors:
        st.warning(e)
    if cluster["errors"]:
        st.error("Cluster settings missing (set them in the Configuration tab):")
        for e in cluster["errors"]:
            st.text(f"- {e}")

    # --- Launch ---
    if st.button("Launch scan", type="primary", disabled=not can_launch, key="satmut_launch"):
        res = satmut.submit_scan(cfg, out_dir, safe, seq, size)
        if res["error"]:
            st.error(f"Submission failed: {res['error']}")
            st.code(res["command"], language="bash")
        else:
            st.success(f"Submitted scan job {res['job_id']} for `{safe}` ({size}). "
                       "Refresh once it finishes to see the heatmap.")

    st.divider()

    # --- Results (works for this launch or any prior scan in the out dir) ---
    st.markdown("##### Result")
    view_name = safe or name
    if not (out_dir and view_name):
        st.info("Enter a name, sequence, and output directory, then launch a scan.")
        return
    if st.button("Refresh result", key="satmut_refresh"):
        st.rerun()

    shown = _heatmap(out_dir, view_name, size)
    if not shown:
        meta_p = satmut.size_dir(out_dir, view_name, size) / "scan_meta.json"
        if meta_p.is_file():
            meta = json.loads(meta_p.read_text())
            st.info(f"Scan job {meta.get('job_id', '?')} submitted "
                    f"{meta.get('submitted', '')[:16]} — logits not present yet. "
                    "Check the Job Monitor; refresh when it completes.")
        else:
            st.info(f"No scan found under `{Path(out_dir) / view_name / size}` yet.")
