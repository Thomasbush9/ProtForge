"""Visualization module for ProtForge.

Provides interactive 3D protein structure plots, per-residue metric
dashboards, and confidence (pLDDT) visualization using Plotly.

Adapted from PDAnalysis visualization module. Works standalone with
the lightweight ``ProteinStructure`` from ``viz._cif_parser``, but
also accepts PDAnalysis Protein/AverageProtein objects via duck typing.

Install dependencies with:
    pip install "protforge[viz]"
    # or simply: pip install plotly
"""

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError(
        "plotly is required for visualization. "
        "Install with: pip install 'protforge[viz]' or pip install plotly"
    )

import numpy as np
import pandas as pd
from pathlib import Path

from ._cif_parser import ProteinStructure, parse_cif


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_valid_coords(protein):
    """Extract CA coordinates and a boolean mask of valid (non-NaN) positions.

    Supports ProteinStructure, PDAnalysis Protein, and AverageProtein
    (detected via duck typing on the ``proteins`` attribute).
    """
    if hasattr(protein, "proteins"):
        # AverageProtein — use first protein's coords
        coords = protein.proteins[0].coord
    else:
        coords = protein.coord
    valid = ~np.any(np.isnan(coords), axis=1)
    return coords, valid


def _get_contiguous_segments(valid_mask):
    """Find contiguous runs of True in *valid_mask*.

    Returns list of (start, end) index pairs (inclusive start, exclusive end).
    """
    segments = []
    in_segment = False
    start = 0
    for i, v in enumerate(valid_mask):
        if v and not in_segment:
            start = i
            in_segment = True
        elif not v and in_segment:
            segments.append((start, i))
            in_segment = False
    if in_segment:
        segments.append((start, len(valid_mask)))
    return segments


def _load_protein(obj, min_plddt=0):
    """Load a protein from a path or return as-is if already an object."""
    if isinstance(obj, (str, Path)):
        return parse_cif(str(obj), min_plddt=min_plddt)
    return obj


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------

_NON_METRIC_COLS = {"residue_index", "protA_resname", "protB_resname"}

_ALL_PANELS = ["heatmap", "profiles", "distributions", "summary"]


def _detect_mutations_from_df(df):
    """Return dict mapping 1-based residue index to mutation string."""
    if "protA_resname" not in df.columns or "protB_resname" not in df.columns:
        return {}
    mask = df["protA_resname"] != df["protB_resname"]
    sub = df[mask]
    mutations = {}
    for _, row in sub.iterrows():
        idx = int(row["residue_index"])
        mutations[idx] = f"{row['protA_resname']}{idx}{row['protB_resname']}"
    return mutations


def _aggregate_mutations(df_dict):
    """Aggregate mutation data across all proteins."""
    per_protein = {}
    freq = {}
    all_positions = set()
    for name, df in df_dict.items():
        muts = _detect_mutations_from_df(df)
        per_protein[name] = muts
        for pos in muts:
            all_positions.add(pos)
            freq[pos] = freq.get(pos, 0) + 1
    return all_positions, freq, per_protein


def _compute_subplot_grid(panels):
    """Return (rows, cols, position_map) for a dynamic panel layout."""
    n = len(panels)
    if n <= 1:
        rows, cols = 1, 1
    elif n == 2:
        rows, cols = 1, 2
    else:
        rows, cols = 2, 2
    positions = {}
    for i, panel in enumerate(panels):
        r = i // cols + 1
        c = i % cols + 1
        positions[panel] = (r, c)
    return rows, cols, positions


def _extract_ref_sequence(df_dict, max_residue):
    """Build a reference amino-acid array from protA_resname columns."""
    ref_aa = [""] * max_residue
    for df in df_dict.values():
        if "protA_resname" not in df.columns:
            continue
        for _, row in df.iterrows():
            idx = int(row["residue_index"]) - 1
            if 0 <= idx < max_residue:
                ref_aa[idx] = str(row["protA_resname"])
        break
    return ref_aa


def _build_aa_customdata(df_dict, names, max_residue, mut_per_protein=None):
    """Build a 2-D list (proteins x residues) of amino-acid hover strings."""
    ref_aa = _extract_ref_sequence(df_dict, max_residue)
    customdata = []
    for name in names:
        muts = (mut_per_protein or {}).get(name, {})
        row = []
        for r in range(max_residue):
            pos = r + 1
            if pos in muts:
                mut_str = muts[pos]
                row.append(f"{mut_str[0]}\u2192{mut_str[-1]}")
            else:
                row.append(ref_aa[r])
        customdata.append(row)
    return customdata, ref_aa


def _detect_metric_columns(df):
    """Return list of numeric column names excluding fixed ID columns."""
    return [
        c for c in df.columns
        if c not in _NON_METRIC_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def _load_df_dict(source):
    """Resolve *source* into dict[str, DataFrame].

    Accepts:
    - dict of {name: DataFrame} -- returned as-is
    - path to a .joblib file -- loaded with joblib.load()
    - path to a directory -- globs *.csv (skipping combined.csv)
    """
    if isinstance(source, dict):
        return source

    source = Path(source)

    if source.suffix == ".joblib":
        import joblib
        data = joblib.load(source)
        if isinstance(data, dict):
            return {k: (v if isinstance(v, pd.DataFrame) else pd.DataFrame(v))
                    for k, v in data.items()}
        raise TypeError(f"Expected dict inside joblib file, got {type(data).__name__}")

    if source.is_dir():
        csvs = sorted(source.glob("*.csv"))
        csvs = [p for p in csvs if p.name != "combined.csv"]
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {source}")
        return {p.stem: pd.read_csv(p) for p in csvs}

    raise ValueError(f"source must be a dict, .joblib path, or directory; got {source}")


def _build_heatmap_matrix(df_dict, metric, max_residue=None):
    """Build a 2-D array (proteins x residues) for the heatmap panel."""
    names = list(df_dict.keys())
    if max_residue is None:
        max_residue = max(
            int(df["residue_index"].max())
            for df in df_dict.values()
            if "residue_index" in df.columns
        )
    matrix = np.full((len(names), max_residue), np.nan)
    for i, name in enumerate(names):
        df = df_dict[name]
        if metric not in df.columns:
            continue
        idx = df["residue_index"].values.astype(int) - 1
        vals = df[metric].values.astype(float)
        valid = (idx >= 0) & (idx < max_residue)
        matrix[i, idx[valid]] = vals[valid]
    return matrix, names


# ---------------------------------------------------------------------------
# Public API — 3D protein plots
# ---------------------------------------------------------------------------

def plot_backbone(protein, **kwargs):
    """3D interactive backbone trace.

    Parameters
    ----------
    protein : ProteinStructure, Protein, AverageProtein, str, or Path
    color, title, point_size, line_width, label, opacity : optional
    """
    protein = _load_protein(protein, kwargs.get("min_plddt", 0))
    color = kwargs.get("color", "royalblue")
    title = kwargs.get("title", "Protein Backbone")
    point_size = kwargs.get("point_size", 3)
    line_width = kwargs.get("line_width", 2)
    label = kwargs.get("label", "backbone")
    opacity = kwargs.get("opacity", 1.0)

    coords, valid = _get_valid_coords(protein)
    segments = _get_contiguous_segments(valid)

    fig = go.Figure()
    for idx, (s, e) in enumerate(segments):
        seg = coords[s:e]
        fig.add_trace(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
            mode="lines+markers",
            marker=dict(size=point_size, color=color, opacity=opacity),
            line=dict(width=line_width, color=color),
            name=label, legendgroup=label, showlegend=(idx == 0),
        ))
    fig.update_layout(title=title, scene=dict(aspectmode="data"), template="plotly_white")
    return fig


def plot_strain(protein, strain_values, **kwargs):
    """3D backbone coloured by per-residue strain.

    Parameters
    ----------
    protein : ProteinStructure, Protein, AverageProtein, str, or Path
    strain_values : array-like, shape (N,)
    cmap, vmin, vmax, colorbar_label, nan_color, title, point_size : optional
    """
    protein = _load_protein(protein, kwargs.get("min_plddt", 0))
    cmap = kwargs.get("cmap", "Viridis")
    colorbar_label = kwargs.get("colorbar_label", "Effective Strain")
    nan_color = kwargs.get("nan_color", "lightgray")
    title = kwargs.get("title", "Strain Map")
    point_size = kwargs.get("point_size", 4)

    coords, valid = _get_valid_coords(protein)
    strain = np.asarray(strain_values, dtype=float)

    finite_mask = valid & np.isfinite(strain)
    vmin = kwargs.get("vmin", float(np.nanmin(strain[finite_mask])) if finite_mask.any() else 0)
    vmax = kwargs.get("vmax", float(np.nanmax(strain[finite_mask])) if finite_mask.any() else 1)

    segments = _get_contiguous_segments(valid)
    fig = go.Figure()

    for s, e in segments:
        seg = coords[s:e]
        fig.add_trace(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
            mode="lines", line=dict(width=1, color="gray"),
            showlegend=False, hoverinfo="skip",
        ))

    nan_strain_mask = valid & ~np.isfinite(strain)
    if nan_strain_mask.any():
        c = coords[nan_strain_mask]
        fig.add_trace(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            mode="markers", marker=dict(size=point_size, color=nan_color, opacity=0.5),
            name="NaN / excluded", showlegend=True,
        ))

    if finite_mask.any():
        c = coords[finite_mask]
        sv = strain[finite_mask]
        residue_idx = np.where(finite_mask)[0]
        hover = [f"Residue {i}<br>Strain: {v:.4f}" for i, v in zip(residue_idx, sv)]
        fig.add_trace(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            mode="markers",
            marker=dict(
                size=point_size, color=sv, colorscale=cmap,
                cmin=vmin, cmax=vmax,
                colorbar=dict(title=colorbar_label), opacity=1.0,
            ),
            text=hover, hoverinfo="text", name="strain", showlegend=False,
        ))

    fig.update_layout(title=title, scene=dict(aspectmode="data"), template="plotly_white")
    return fig


def plot_confidence(protein, **kwargs):
    """3D backbone coloured by per-residue pLDDT confidence.

    Parameters
    ----------
    protein : ProteinStructure, Protein, AverageProtein, str, or Path
        Must have a ``plddt`` attribute.
    cmap : str, optional
        Plotly colour scale (default ``"RdYlGn"`` — red=low, green=high).
    vmin, vmax : float, optional
        Colour scale bounds (default 0–100).
    title : str, optional
    point_size : float, optional

    Returns
    -------
    plotly.graph_objects.Figure
    """
    protein = _load_protein(protein, kwargs.get("min_plddt", 0))
    cmap = kwargs.get("cmap", "RdYlGn")
    vmin = kwargs.get("vmin", 0)
    vmax = kwargs.get("vmax", 100)
    title = kwargs.get("title", "Model Confidence (pLDDT)")
    point_size = kwargs.get("point_size", 4)
    nan_color = kwargs.get("nan_color", "lightgray")

    coords, valid = _get_valid_coords(protein)
    plddt = np.asarray(protein.plddt, dtype=float)

    segments = _get_contiguous_segments(valid)
    fig = go.Figure()

    # Backbone lines
    for s, e in segments:
        seg = coords[s:e]
        fig.add_trace(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
            mode="lines", line=dict(width=1, color="gray"),
            showlegend=False, hoverinfo="skip",
        ))

    # Confidence-coloured markers
    finite_mask = valid & np.isfinite(plddt)
    if finite_mask.any():
        c = coords[finite_mask]
        pv = plddt[finite_mask]
        residue_idx = np.where(finite_mask)[0]
        hover = [f"Residue {i+1}<br>pLDDT: {v:.1f}" for i, v in zip(residue_idx, pv)]
        fig.add_trace(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            mode="markers",
            marker=dict(
                size=point_size, color=pv, colorscale=cmap,
                cmin=vmin, cmax=vmax,
                colorbar=dict(title="pLDDT"), opacity=1.0,
            ),
            text=hover, hoverinfo="text", name="confidence", showlegend=False,
        ))

    # Low-confidence / NaN markers
    nan_mask = valid & ~np.isfinite(plddt)
    if nan_mask.any():
        c = coords[nan_mask]
        fig.add_trace(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            mode="markers", marker=dict(size=point_size, color=nan_color, opacity=0.5),
            name="NaN / excluded", showlegend=True,
        ))

    fig.update_layout(title=title, scene=dict(aspectmode="data"), template="plotly_white")
    return fig


def plot_comparison(proteinA, proteinB, **kwargs):
    """Overlay two backbones, optionally Kabsch-aligned.

    When *strain_values* is provided both backbones are coloured by strain.
    """
    min_plddt = kwargs.get("min_plddt", 0)
    proteinA = _load_protein(proteinA, min_plddt)
    proteinB = _load_protein(proteinB, min_plddt)

    color_A = kwargs.get("color_A", "royalblue")
    color_B = kwargs.get("color_B", "crimson")
    label_A = kwargs.get("label_A", "Protein A")
    label_B = kwargs.get("label_B", "Protein B")
    align = kwargs.get("align", True)
    title = kwargs.get("title", "Structure Comparison")
    point_size = kwargs.get("point_size", 3)
    cmap = kwargs.get("cmap", "Viridis")
    colorbar_label = kwargs.get("colorbar_label", "Effective Strain")

    strain = None
    strain_values = kwargs.get("strain_values", None)
    if strain_values is not None:
        strain = np.asarray(strain_values, dtype=float)

    coordsA, validA = _get_valid_coords(proteinA)
    coordsB, validB = _get_valid_coords(proteinB)

    if align:
        shared_valid = validA & validB
        if shared_valid.any():
            cA = coordsA[shared_valid]
            cB = coordsB[shared_valid]
            cenA = cA.mean(axis=0)
            cenB = cB.mean(axis=0)
            H = (cB - cenB).T @ (cA - cenA)
            U, S, Vt = np.linalg.svd(H)
            V = Vt.T
            D = np.linalg.det(V @ U.T)
            E = np.diag([1, 1, D])
            R = V @ E @ U.T
            coordsB = ((R @ (coordsB - cenB).T).T) + cenA

    if strain is not None:
        finite_mask = (validA | validB) & np.isfinite(strain)
        vmin = float(np.nanmin(strain[finite_mask])) if finite_mask.any() else 0
        vmax = float(np.nanmax(strain[finite_mask])) if finite_mask.any() else 1

    fig = go.Figure()
    shown_colorbar = False
    for prot_coords, valid, line_color, label in [
        (coordsA, validA, color_A, label_A),
        (coordsB, validB, color_B, label_B),
    ]:
        segments = _get_contiguous_segments(valid)
        for idx, (s, e) in enumerate(segments):
            seg = prot_coords[s:e]
            fig.add_trace(go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines", line=dict(width=2, color=line_color),
                name=label, legendgroup=label, showlegend=(idx == 0),
            ))

        if strain is not None:
            finite = valid & np.isfinite(strain)
            if finite.any():
                c = prot_coords[finite]
                sv = strain[finite]
                residue_idx = np.where(finite)[0]
                hover = [f"Residue {i+1}<br>Strain: {v:.4f}" for i, v in zip(residue_idx, sv)]
                marker_dict = dict(
                    size=point_size, color=sv, colorscale=cmap,
                    cmin=vmin, cmax=vmax, opacity=1.0,
                )
                if not shown_colorbar:
                    marker_dict["colorbar"] = dict(title=colorbar_label)
                    shown_colorbar = True
                fig.add_trace(go.Scatter3d(
                    x=c[:, 0], y=c[:, 1], z=c[:, 2],
                    mode="markers", marker=marker_dict,
                    text=hover, hoverinfo="text",
                    name=f"{label} strain", legendgroup=label, showlegend=False,
                ))
        else:
            valid_coords = prot_coords[valid]
            residue_idx = np.where(valid)[0]
            hover = [f"Residue {i+1}" for i in residue_idx]
            fig.add_trace(go.Scatter3d(
                x=valid_coords[:, 0], y=valid_coords[:, 1], z=valid_coords[:, 2],
                mode="markers", marker=dict(size=point_size, color=line_color),
                text=hover, hoverinfo="text",
                name=label, legendgroup=label, showlegend=False,
            ))

    fig.update_layout(title=title, scene=dict(aspectmode="data"), template="plotly_white")
    return fig


# ---------------------------------------------------------------------------
# Multi-protein dashboard
# ---------------------------------------------------------------------------

def get_available_metrics(source):
    """Return sorted list of metric columns common to all DataFrames."""
    df_dict = _load_df_dict(source)
    sets = [set(_detect_metric_columns(df)) for df in df_dict.values()]
    common = sets[0]
    for s in sets[1:]:
        common &= s
    return sorted(common)


def plot_dashboard(source, **kwargs):
    """Interactive dashboard comparing N proteins on a single metric.

    Produces a 2x2 grid (heatmap, profiles, distributions, summary).

    Parameters
    ----------
    source : dict[str, DataFrame] | str | Path
        Anything accepted by ``_load_df_dict``.
    metric : str
        Column name to visualise (default ``"strain"``).
    sort_by : str or None
        Sort by ``"mean"``, ``"max"``, ``"median"``, or None.
    panels : list[str] or None
        Subset of ``["heatmap", "profiles", "distributions", "summary"]``.
    top_n : int or None
    proteins : list[str] or None
    highlight_mutations : bool
    mutation_color : str
    log_y : bool
    cmap : str
    line_opacity : float or None
    show_violin : bool
    title, height, width : optional
    truncate_labels : int or None
    vmin, vmax : float or None
    """
    import plotly.express as px

    df_dict = _load_df_dict(source)
    metric = kwargs.get("metric", "strain")
    sort_by = kwargs.get("sort_by", "mean")
    cmap = kwargs.get("cmap", "Viridis")
    line_opacity = kwargs.get("line_opacity", None)
    show_violin = kwargs.get("show_violin", True)
    title = kwargs.get("title", None)
    height = kwargs.get("height", None)
    width = kwargs.get("width", None)
    truncate_labels = kwargs.get("truncate_labels", 30)
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    panels = kwargs.get("panels", None)
    top_n = kwargs.get("top_n", None)
    highlight_mutations = kwargs.get("highlight_mutations", False)
    mutation_color = kwargs.get("mutation_color", "rgba(255,0,0,0.4)")
    log_y = kwargs.get("log_y", False)
    proteins = kwargs.get("proteins", None)

    if panels is None:
        panels = list(_ALL_PANELS)
    for p in panels:
        if p not in _ALL_PANELS:
            raise ValueError(f"Unknown panel '{p}'. Choose from {_ALL_PANELS}")

    if proteins is not None:
        df_dict = {n: df_dict[n] for n in proteins if n in df_dict}

    N = len(df_dict)
    if N == 0:
        raise ValueError("source contains no protein data")

    # Compute summary stats
    stats = {}
    for name, df in df_dict.items():
        vals = df[metric].dropna().values.astype(float) if metric in df.columns else np.array([])
        stats[name] = {
            "mean": float(np.nanmean(vals)) if len(vals) else 0.0,
            "max": float(np.nanmax(vals)) if len(vals) else 0.0,
            "median": float(np.nanmedian(vals)) if len(vals) else 0.0,
        }

    names = list(df_dict.keys())
    if sort_by in ("mean", "max", "median"):
        names = sorted(names, key=lambda n: stats[n][sort_by], reverse=True)

    if top_n is not None and top_n < len(names):
        names = names[:top_n]

    df_dict = {n: df_dict[n] for n in names}
    N = len(df_dict)

    # Mutation data
    mut_positions = set()
    mut_freq = {}
    mut_per_protein = {}
    if highlight_mutations:
        mut_positions, mut_freq, mut_per_protein = _aggregate_mutations(df_dict)
        max_freq = max(mut_freq.values()) if mut_freq else 1

    if truncate_labels:
        display_names = [
            n[:truncate_labels] + "\u2026" if len(n) > truncate_labels else n
            for n in names
        ]
    else:
        display_names = list(names)

    rows, cols, pos_map = _compute_subplot_grid(panels)

    if height is None:
        base = 500 if rows == 1 else 800
        height = min(max(base, 120 + N * 18), 2000)

    palette = px.colors.qualitative.Plotly
    colours = [palette[i % len(palette)] for i in range(N)]

    if line_opacity is None:
        line_opacity = max(0.15, 1.0 - N * 0.03)

    ref_aa = []

    subplot_titles = [
        {"heatmap": f"Heatmap \u2014 {metric}",
         "profiles": f"Per-Residue Profiles \u2014 {metric}",
         "distributions": f"Distributions \u2014 {metric}",
         "summary": f"Summary \u2014 {metric}"}[p]
        for p in panels
    ]
    while len(subplot_titles) < rows * cols:
        subplot_titles.append("")

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # --- Heatmap ---
    if "heatmap" in pos_map:
        r, c = pos_map["heatmap"]
        matrix, _ = _build_heatmap_matrix(df_dict, metric)
        n_residues = matrix.shape[1]
        aa_customdata, ref_aa = _build_aa_customdata(
            df_dict, names, n_residues, mut_per_protein if highlight_mutations else None
        )

        if log_y:
            raw_matrix = matrix.copy()
            with np.errstate(divide="ignore", invalid="ignore"):
                z_display = np.where(
                    np.isfinite(matrix) & (matrix > 0),
                    np.log10(matrix), np.nan,
                )
            raw_list = raw_matrix.tolist()
            combined_cd = [
                [{"aa": aa_customdata[i][j], "v": raw_list[i][j]}
                 for j in range(n_residues)]
                for i in range(len(names))
            ]
            finite_z = z_display[np.isfinite(z_display)]
            if finite_z.size:
                zlo, zhi = float(finite_z.min()), float(finite_z.max())
            else:
                zlo, zhi = -3, 0
            import math
            tick_exp = list(range(math.floor(zlo), math.ceil(zhi) + 1))
            cb = dict(
                title=f"{metric} (log)", x=0.45, len=0.45, y=0.78,
                tickvals=tick_exp,
                ticktext=[f"{10**e:.3g}" for e in tick_exp],
            )
            fig.add_trace(go.Heatmap(
                z=z_display, x=np.arange(1, n_residues + 1), y=display_names,
                colorscale=cmap,
                zmin=vmin if vmin is not None else zlo,
                zmax=vmax if vmax is not None else zhi,
                colorbar=cb, customdata=combined_cd,
                hovertemplate=(
                    "Residue %{x} (%{customdata.aa})<br>"
                    "%{y}<br>Value: %{customdata.v:.4f}<extra></extra>"
                ),
            ), row=r, col=c)
        else:
            fig.add_trace(go.Heatmap(
                z=matrix, x=np.arange(1, n_residues + 1), y=display_names,
                colorscale=cmap, zmin=vmin, zmax=vmax,
                colorbar=dict(title=metric, x=0.45, len=0.45, y=0.78),
                customdata=aa_customdata,
                hovertemplate="Residue %{x} (%{customdata})<br>%{y}<br>Value: %{z:.4f}<extra></extra>",
            ), row=r, col=c)

        if highlight_mutations and mut_positions:
            for pos, freq in mut_freq.items():
                opacity = 0.15 + 0.55 * (freq / max_freq)
                fig.add_vrect(
                    x0=pos - 0.5, x1=pos + 0.5,
                    fillcolor=mutation_color, opacity=opacity, line_width=0,
                    row=r, col=c,
                )

    # --- Profiles ---
    if "profiles" in pos_map:
        r, c = pos_map["profiles"]
        if not ref_aa:
            _any_df = next(iter(df_dict.values()))
            _max_res = int(_any_df["residue_index"].max()) if "residue_index" in _any_df.columns else 0
            ref_aa = _extract_ref_sequence(df_dict, _max_res) if _max_res else []
        for i, name in enumerate(names):
            df = df_dict[name]
            if metric not in df.columns:
                continue
            res_idx = df["residue_index"].values.astype(int)
            muts = mut_per_protein.get(name, {}) if highlight_mutations else {}
            hover_text = []
            for ri in res_idx:
                aa_idx = ri - 1
                if ri in muts:
                    aa_str = muts[ri]
                elif 0 <= aa_idx < len(ref_aa):
                    aa_str = ref_aa[aa_idx]
                else:
                    aa_str = ""
                hover_text.append(f"Res {ri} ({aa_str})")
            fig.add_trace(go.Scatter(
                x=res_idx, y=df[metric], mode="lines",
                line=dict(color=colours[i], width=1.2), opacity=line_opacity,
                name=display_names[i], legendgroup=display_names[i], showlegend=True,
                text=hover_text,
                hovertemplate="%{text}<br>%{y:.4f}<extra>%{fullData.name}</extra>",
            ), row=r, col=c)
        if highlight_mutations and mut_positions:
            for pos in sorted(mut_positions):
                fig.add_vline(
                    x=pos, line_dash="dash", line_color=mutation_color,
                    opacity=0.6, row=r, col=c,
                )

    # --- Distributions ---
    if "distributions" in pos_map:
        r, c = pos_map["distributions"]
        for i, name in enumerate(names):
            df = df_dict[name]
            if metric not in df.columns:
                continue
            vals = df[metric].dropna().values
            if show_violin:
                fig.add_trace(go.Violin(
                    y=vals, name=display_names[i], legendgroup=display_names[i],
                    showlegend=False, line_color=colours[i],
                    meanline_visible=True, scalemode="width", width=0.8,
                ), row=r, col=c)
            else:
                fig.add_trace(go.Box(
                    y=vals, name=display_names[i], legendgroup=display_names[i],
                    showlegend=False, marker_color=colours[i],
                ), row=r, col=c)

    # --- Summary bar ---
    if "summary" in pos_map:
        r, c = pos_map["summary"]
        mean_vals = [stats[n]["mean"] for n in names]
        max_vals = [stats[n]["max"] for n in names]

        hover_texts = []
        for i, name in enumerate(names):
            parts = [
                f"<b>{display_names[i]}</b>",
                f"Mean: {mean_vals[i]:.4f}",
                f"Max: {max_vals[i]:.4f}",
                f"Median: {stats[name]['median']:.4f}",
            ]
            if highlight_mutations and mut_per_protein:
                muts = mut_per_protein.get(name, {})
                if muts:
                    parts.append(f"Mutations: {', '.join(str(v) for v in muts.values())}")
            hover_texts.append("<br>".join(parts))

        fig.add_trace(go.Bar(
            x=display_names, y=mean_vals, marker_color=colours,
            name="Mean", showlegend=False,
            hovertemplate="%{customdata}<extra></extra>", customdata=hover_texts,
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=display_names, y=max_vals, mode="markers",
            marker=dict(symbol="diamond", size=9, color=colours,
                        line=dict(width=1, color="black")),
            name="Max", showlegend=False,
            hovertemplate="%{x}<br>Max: %{y:.4f}<extra></extra>",
        ), row=r, col=c)

    # --- Layout polish ---
    for panel, (r, c) in pos_map.items():
        if panel in ("heatmap", "profiles"):
            fig.update_xaxes(title_text="Residue Index", row=r, col=c)
        if panel in ("profiles", "distributions"):
            fig.update_yaxes(title_text=metric, row=r, col=c,
                             type="log" if log_y else "linear")
        if panel == "summary":
            fig.update_xaxes(title_text="Protein", tickangle=45, row=r, col=c)
            fig.update_yaxes(title_text=metric, row=r, col=c,
                             type="log" if log_y else "linear")

    dashboard_title = title or f"Multi-Protein Dashboard \u2014 {metric}"
    fig.update_layout(
        title=dashboard_title, height=height, width=width,
        autosize=True, template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=0.45, xanchor="left", x=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# 3D overlay with mutation markers
# ---------------------------------------------------------------------------

def plot_3d_overlay(proteinA, proteinB, **kwargs):
    """3D backbone overlay with optional strain colouring and mutation markers.

    Accepts ProteinStructure, PDAnalysis objects, or file paths.
    """
    min_plddt = kwargs.get("min_plddt", 0)
    proteinA = _load_protein(proteinA, min_plddt)
    proteinB = _load_protein(proteinB, min_plddt)

    color_A = kwargs.get("color_A", "royalblue")
    color_B = kwargs.get("color_B", "crimson")
    label_A = kwargs.get("label_A", "Protein A")
    label_B = kwargs.get("label_B", "Protein B")
    align = kwargs.get("align", True)
    title = kwargs.get("title", "3D Structure Overlay")
    point_size = kwargs.get("point_size", 3)
    cmap = kwargs.get("cmap", "Viridis")
    colorbar_label = kwargs.get("colorbar_label", "Effective Strain")
    mutation_marker_size = kwargs.get("mutation_marker_size", 8)
    mutation_symbol = kwargs.get("mutation_symbol", "diamond")

    strain = None
    strain_values = kwargs.get("strain_values", None)
    if strain_values is not None:
        strain = np.asarray(strain_values, dtype=float)

    # Auto-detect mutations from sequences
    mutations = kwargs.get("mutations", None)
    if mutations is None:
        seqA = getattr(proteinA, "sequence", None)
        seqB = getattr(proteinB, "sequence", None)
        if seqA is not None and seqB is not None:
            mutations = {}
            for i, (a, b) in enumerate(zip(seqA, seqB)):
                if a != b:
                    mutations[i + 1] = f"{a}{i+1}{b}"

    coordsA, validA = _get_valid_coords(proteinA)
    coordsB, validB = _get_valid_coords(proteinB)

    if align:
        shared_valid = validA & validB
        if shared_valid.any():
            cA = coordsA[shared_valid]
            cB = coordsB[shared_valid]
            cenA = cA.mean(axis=0)
            cenB = cB.mean(axis=0)
            H = (cB - cenB).T @ (cA - cenA)
            U, S, Vt = np.linalg.svd(H)
            V = Vt.T
            D = np.linalg.det(V @ U.T)
            E = np.diag([1, 1, D])
            R = V @ E @ U.T
            coordsB = ((R @ (coordsB - cenB).T).T) + cenA

    if strain is not None:
        finite_mask = (validA | validB) & np.isfinite(strain)
        vmin = float(np.nanmin(strain[finite_mask])) if finite_mask.any() else 0
        vmax = float(np.nanmax(strain[finite_mask])) if finite_mask.any() else 1

    fig = go.Figure()
    shown_colorbar = False

    for prot_coords, valid, line_color, label in [
        (coordsA, validA, color_A, label_A),
        (coordsB, validB, color_B, label_B),
    ]:
        segments = _get_contiguous_segments(valid)
        for idx, (s, e) in enumerate(segments):
            seg = prot_coords[s:e]
            fig.add_trace(go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines", line=dict(width=2, color=line_color),
                name=label, legendgroup=label, showlegend=(idx == 0),
            ))

        if strain is not None:
            finite = valid & np.isfinite(strain)
            if finite.any():
                c = prot_coords[finite]
                sv = strain[finite]
                residue_idx = np.where(finite)[0]
                hover = [f"Residue {i+1}<br>Strain: {v:.4f}" for i, v in zip(residue_idx, sv)]
                marker_dict = dict(
                    size=point_size, color=sv, colorscale=cmap,
                    cmin=vmin, cmax=vmax, opacity=1.0,
                )
                if not shown_colorbar:
                    marker_dict["colorbar"] = dict(title=colorbar_label)
                    shown_colorbar = True
                fig.add_trace(go.Scatter3d(
                    x=c[:, 0], y=c[:, 1], z=c[:, 2],
                    mode="markers", marker=marker_dict,
                    text=hover, hoverinfo="text",
                    name=f"{label} strain", legendgroup=label, showlegend=False,
                ))
        else:
            valid_coords = prot_coords[valid]
            residue_idx = np.where(valid)[0]
            hover = [f"Residue {i+1}" for i in residue_idx]
            fig.add_trace(go.Scatter3d(
                x=valid_coords[:, 0], y=valid_coords[:, 1], z=valid_coords[:, 2],
                mode="markers", marker=dict(size=point_size, color=line_color),
                text=hover, hoverinfo="text",
                name=label, legendgroup=label, showlegend=False,
            ))

    # Mutation markers
    if mutations:
        mut_x, mut_y, mut_z, mut_text = [], [], [], []
        for pos, label_str in mutations.items():
            idx = pos - 1
            if idx < len(coordsA) and validA[idx]:
                mut_x.append(coordsA[idx, 0])
                mut_y.append(coordsA[idx, 1])
                mut_z.append(coordsA[idx, 2])
                mut_text.append(label_str)
        if mut_x:
            fig.add_trace(go.Scatter3d(
                x=mut_x, y=mut_y, z=mut_z, mode="markers",
                marker=dict(symbol=mutation_symbol, size=mutation_marker_size,
                            color="red", line=dict(width=2, color="black"), opacity=1.0),
                text=mut_text, hoverinfo="text", name="Mutations", showlegend=True,
            ))

    fig.update_layout(title=title, scene=dict(aspectmode="data"), template="plotly_white")
    return fig


def _df_metric_to_array(df, metric_col, n_residues):
    """Convert a DataFrame metric column to a per-residue array (0-indexed)."""
    arr = np.full(n_residues, np.nan)
    if df is not None and metric_col in df.columns and "residue_index" in df.columns:
        idx = df["residue_index"].values.astype(int) - 1
        vals = df[metric_col].values.astype(float)
        valid = (idx >= 0) & (idx < n_residues)
        arr[idx[valid]] = vals[valid]
    return arr


def plot_3d_overlay_multi(reference, targets, **kwargs):
    """3D overlay with a dropdown to switch between target proteins.

    When *confidence_dict* is provided alongside *df_dict*, a second
    dropdown allows switching the marker colouring between the ES metric
    (e.g. strain) and model confidence (pLDDT).

    Parameters
    ----------
    reference : ProteinStructure, str, or Path
    targets : dict[str, ProteinStructure | str | Path]
    df_dict : dict[str, DataFrame] or None
        Per-protein ES CSVs (strain etc.).
    confidence_dict : dict[str, DataFrame] or None
        Per-protein pLDDT DataFrames (from ``load_confidence_dfs``).
    metric : str
        Column in *df_dict* to use for strain colouring (default ``"strain"``).
    """
    import plotly.express as px

    min_plddt = kwargs.get("min_plddt", 0)
    color_A = kwargs.get("color_A", "royalblue")
    label_A = kwargs.get("label_A", "Reference")
    align = kwargs.get("align", True)
    title = kwargs.get("title", "3D Multi-Overlay")
    point_size = kwargs.get("point_size", 3)
    cmap = kwargs.get("cmap", "Viridis")
    colorbar_label = kwargs.get("colorbar_label", "Strain")
    mutation_marker_size = kwargs.get("mutation_marker_size", 8)
    mutation_symbol = kwargs.get("mutation_symbol", "diamond")
    df_dict = kwargs.get("df_dict", None)
    confidence_dict = kwargs.get("confidence_dict", None)
    metric = kwargs.get("metric", "strain")
    palette = px.colors.qualitative.Plotly

    has_strain = df_dict is not None
    has_confidence = confidence_dict is not None
    dual_metric = has_strain and has_confidence

    reference = _load_protein(reference, min_plddt)
    coordsA, validA = _get_valid_coords(reference)
    n_ref = len(coordsA)
    seqA = getattr(reference, "sequence", None)

    fig = go.Figure()

    # Reference backbone (always visible)
    segments = _get_contiguous_segments(validA)
    for seg_i, (s, e) in enumerate(segments):
        seg = coordsA[s:e]
        fig.add_trace(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
            mode="lines", line=dict(width=2, color=color_A),
            name=label_A, legendgroup="ref", showlegend=(seg_i == 0),
        ))
    ref_trace_count = len(fig.data)

    # Per-target traces
    target_names = list(targets.keys())
    trace_ranges = {}
    # For dual-metric restyle: track marker trace indices and their data
    metric_trace_indices = []
    metric_stored_data = {}  # {trace_idx: {"strain": vals, "plddt": vals, "res_idx": indices}}

    for t_i, (tname, target) in enumerate(targets.items()):
        start_idx = len(fig.data)
        tcolor = palette[(t_i + 1) % len(palette)]

        target = _load_protein(target, min_plddt)
        coordsB, validB = _get_valid_coords(target)

        if align:
            shared_valid = validA & validB
            if shared_valid.any():
                cA = coordsA[shared_valid]
                cB = coordsB[shared_valid]
                cenA = cA.mean(axis=0)
                cenB = cB.mean(axis=0)
                H = (cB - cenB).T @ (cA - cenA)
                U, S, Vt = np.linalg.svd(H)
                V = Vt.T
                D = np.linalg.det(V @ U.T)
                E = np.diag([1, 1, D])
                R = V @ E @ U.T
                coordsB = ((R @ (coordsB - cenB).T).T) + cenA

        # Build per-residue arrays for each metric
        strain_arr = _df_metric_to_array(
            df_dict.get(tname) if df_dict else None, metric, n_ref
        )
        plddt_arr = _df_metric_to_array(
            confidence_dict.get(tname) if confidence_dict else None, "plddt", n_ref
        )

        # Determine which positions have ANY metric data
        has_any = validA & (np.isfinite(strain_arr) | np.isfinite(plddt_arr))

        # Mutations
        mutations = {}
        if df_dict and tname in df_dict:
            mutations = _detect_mutations_from_df(df_dict[tname])
        elif seqA is not None:
            seqB = getattr(target, "sequence", None)
            if seqB is not None:
                for i, (a, b) in enumerate(zip(seqA, seqB)):
                    if a != b:
                        mutations[i + 1] = f"{a}{i+1}{b}"

        # Target backbone lines
        segB = _get_contiguous_segments(validB)
        for seg_i, (s, e) in enumerate(segB):
            seg = coordsB[s:e]
            fig.add_trace(go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines", line=dict(width=1.5, color=tcolor), opacity=0.4,
                name=tname, legendgroup=tname, showlegend=(seg_i == 0),
            ))

        # Metric markers (at reference positions)
        if has_any.any():
            c = coordsA[has_any]
            residue_idx = np.where(has_any)[0]
            sv = strain_arr[has_any]
            pv = plddt_arr[has_any]

            # Choose initial display: strain if available, else pLDDT
            if has_strain and np.isfinite(sv).any():
                init_vals = sv
                init_cmap = cmap
                init_label = colorbar_label
                init_hover = [
                    f"Res {i+1}<br>Strain: {v:.4f}" for i, v in zip(residue_idx, sv)
                ]
            elif has_confidence and np.isfinite(pv).any():
                init_vals = pv
                init_cmap = "RdYlGn"
                init_label = "pLDDT"
                init_hover = [
                    f"Res {i+1}<br>pLDDT: {v:.1f}" for i, v in zip(residue_idx, pv)
                ]
            else:
                init_vals = sv
                init_cmap = cmap
                init_label = colorbar_label
                init_hover = [f"Res {i+1}" for i in residue_idx]

            finite = np.isfinite(init_vals)
            fmin = float(np.nanmin(init_vals[finite])) if finite.any() else 0
            fmax = float(np.nanmax(init_vals[finite])) if finite.any() else 1

            fig.add_trace(go.Scatter3d(
                x=c[:, 0], y=c[:, 1], z=c[:, 2],
                mode="markers",
                marker=dict(
                    size=point_size, color=init_vals.tolist(),
                    colorscale=init_cmap,
                    cmin=fmin, cmax=fmax,
                    colorbar=dict(title=init_label), opacity=1.0,
                ),
                text=init_hover, hoverinfo="text",
                name=f"{tname} metric", legendgroup=tname, showlegend=False,
            ))

            # Store data for metric restyle
            trace_idx = len(fig.data) - 1
            metric_trace_indices.append(trace_idx)
            metric_stored_data[trace_idx] = {
                "strain": sv,
                "plddt": pv,
                "res_idx": residue_idx,
            }

        # Mutation markers
        if mutations:
            mx, my, mz, mt = [], [], [], []
            for pos, lbl in mutations.items():
                idx = pos - 1
                if idx < len(coordsA) and validA[idx]:
                    mx.append(coordsA[idx, 0])
                    my.append(coordsA[idx, 1])
                    mz.append(coordsA[idx, 2])
                    mt.append(lbl)
            if mx:
                fig.add_trace(go.Scatter3d(
                    x=mx, y=my, z=mz, mode="markers",
                    marker=dict(symbol=mutation_symbol, size=mutation_marker_size,
                                color="red", line=dict(width=2, color="black")),
                    text=mt, hoverinfo="text",
                    name=f"{tname} mutations", legendgroup=tname, showlegend=False,
                ))

        trace_ranges[tname] = (start_idx, len(fig.data))

    # --- Dropdown 1: switch target ---
    target_buttons = []
    for tname in target_names:
        visible = [True] * ref_trace_count
        for other_name in target_names:
            s, e = trace_ranges[other_name]
            for _ in range(s, e):
                visible.append(other_name == tname)
        target_buttons.append(dict(method="update", label=tname, args=[{"visible": visible}]))

    if target_names:
        first = target_names[0]
        for tname, (s, e) in trace_ranges.items():
            for idx in range(s, e):
                fig.data[idx].visible = (tname == first)

    # --- Dropdown 2: switch metric (only when both strain and pLDDT available) ---
    menus = [dict(
        type="dropdown", direction="down", buttons=target_buttons,
        x=0.02, xanchor="left", y=1.08, yanchor="top",
        showactive=True, bgcolor="white", bordercolor="#ccc",
    )]

    if dual_metric and metric_trace_indices:
        metric_options = [
            (colorbar_label, "strain", cmap),
            ("Confidence (pLDDT)", "plddt", "RdYlGn"),
        ]

        metric_buttons = []
        for m_label, m_key, m_colorscale in metric_options:
            colors = []
            cmins = []
            cmaxs = []
            texts = []
            for tidx in metric_trace_indices:
                vals = metric_stored_data[tidx][m_key]
                colors.append(vals.tolist())
                finite = vals[np.isfinite(vals)]
                cmins.append(float(finite.min()) if len(finite) else 0)
                cmaxs.append(float(finite.max()) if len(finite) else 1)
                res_idx = metric_stored_data[tidx]["res_idx"]
                if m_key == "plddt":
                    hover = [f"Res {i+1}<br>pLDDT: {v:.1f}" for i, v in zip(res_idx, vals)]
                else:
                    hover = [f"Res {i+1}<br>{m_label}: {v:.4f}" for i, v in zip(res_idx, vals)]
                texts.append(hover)

            metric_buttons.append(dict(
                method="restyle",
                label=m_label,
                args=[{
                    "marker.color": colors,
                    "marker.cmin": cmins,
                    "marker.cmax": cmaxs,
                    "marker.colorscale": [m_colorscale] * len(metric_trace_indices),
                    "marker.colorbar.title": [m_label] * len(metric_trace_indices),
                    "text": texts,
                }, metric_trace_indices],
            ))

        menus.append(dict(
            type="buttons", direction="left", buttons=metric_buttons,
            x=0.02, xanchor="left", y=1.16, yanchor="top",
            showactive=True, bgcolor="white", bordercolor="#ccc",
        ))

    fig.update_layout(
        updatemenus=menus,
        title=title, scene=dict(aspectmode="data"),
        template="plotly_white", autosize=True,
        height=kwargs.get("height", 850),
        margin=dict(l=0, r=0, t=70 if dual_metric else 50, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Tabbed HTML export
# ---------------------------------------------------------------------------

def write_dashboard_html(figures, output, **kwargs):
    """Write multiple Plotly figures into a single tabbed HTML file.

    Parameters
    ----------
    figures : dict[str, go.Figure]
        Mapping of tab label to Plotly figure.
    output : str or Path
    title : str, optional
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    page_title = kwargs.get("title", "ProtForge Dashboard")

    tab_names = list(figures.keys())
    tab_divs = []
    for i, (name, fig) in enumerate(figures.items()):
        div_html = fig.to_html(include_plotlyjs=(i == 0), full_html=False)
        display = "block" if i == 0 else "none"
        safe_id = f"tab-{i}"
        tab_divs.append(
            f'<div id="{safe_id}" class="tab-content" style="display:{display};">'
            f'{div_html}</div>'
        )

    buttons = []
    for i, name in enumerate(tab_names):
        active = ' class="active"' if i == 0 else ""
        buttons.append(f'<button{active} onclick="switchTab({i})">{name}</button>')
    button_bar = '<div class="tab-bar">' + "".join(buttons) + "</div>"

    js = """
<script>
function switchTab(idx) {
    var tabs = document.querySelectorAll('.tab-content');
    var buttons = document.querySelectorAll('.tab-bar button');
    for (var i = 0; i < tabs.length; i++) {
        tabs[i].style.display = (i === idx) ? 'block' : 'none';
        buttons[i].className = (i === idx) ? 'active' : '';
    }
    var visibleTab = tabs[idx];
    var plots = visibleTab.querySelectorAll('.plotly-graph-div');
    for (var j = 0; j < plots.length; j++) {
        if (window.Plotly) { window.Plotly.Plots.resize(plots[j]); }
    }
}
window.addEventListener('resize', function() {
    var plots = document.querySelectorAll('.tab-content[style*="block"] .plotly-graph-div');
    for (var i = 0; i < plots.length; i++) {
        if (window.Plotly) { window.Plotly.Plots.resize(plots[i]); }
    }
});
</script>
"""

    css = """
<style>
html, body { height: 100%; margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
body { display: flex; flex-direction: column; }
.tab-bar { background: #f8f9fa; border-bottom: 2px solid #dee2e6; padding: 8px 16px 0;
    display: flex; gap: 4px; flex-shrink: 0; }
.tab-bar button { padding: 8px 20px; border: 1px solid #dee2e6; border-bottom: none;
    border-radius: 6px 6px 0 0; background: #e9ecef; cursor: pointer;
    font-size: 14px; font-weight: 500; color: #495057; transition: all 0.15s; }
.tab-bar button:hover { background: #fff; }
.tab-bar button.active { background: #fff; border-bottom: 2px solid #fff;
    margin-bottom: -2px; color: #212529; font-weight: 600; }
.tab-content { flex: 1; padding: 0; overflow: auto; }
.tab-content .plotly-graph-div { width: 100% !important; }
</style>
"""

    html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{page_title}</title>{css}</head><body>"
        f"{button_bar}{''.join(tab_divs)}{js}</body></html>"
    )
    output.write_text(html)
    return str(output)
