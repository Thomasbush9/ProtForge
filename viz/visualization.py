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


def _generate_palette(n):
    """Generate *n* visually distinct colors using golden-ratio hue sampling.

    Falls back to the built-in Plotly qualitative palette for n <= 10.
    """
    if n <= 10:
        import plotly.express as px
        base = px.colors.qualitative.Plotly
        return [base[i % len(base)] for i in range(n)]
    colors = []
    golden = 0.618033988749895
    hue = 0.0
    for i in range(n):
        hue = (hue + golden) % 1.0
        sat = 65 + (i % 3) * 12   # 65, 77, 89
        lit = 45 + (i % 4) * 8    # 45, 53, 61, 69
        colors.append(f"hsl({hue * 360:.0f}, {sat}%, {lit}%)")
    return colors


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
    """Return (rows, cols, position_map, specs, row_heights) for dynamic layout.

    When all 4 panels are present, uses an asymmetric layout:
      Row 1: Heatmap (full width, 50%)
      Row 2: Profiles | Distributions (30%)
      Row 3: Summary (full width, 20%)
    """
    n = len(panels)
    panel_set = set(panels)

    if n == 4 and panel_set == set(_ALL_PANELS):
        rows, cols = 3, 2
        positions = {
            "heatmap": (1, 1),
            "profiles": (2, 1),
            "distributions": (2, 2),
            "summary": (3, 1),
        }
        specs = [
            [{"colspan": 2}, None],
            [{}, {}],
            [{"colspan": 2}, None],
        ]
        row_heights = [0.5, 0.3, 0.2]
        return rows, cols, positions, specs, row_heights

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
    specs = [[{} for _ in range(cols)] for _ in range(rows)]
    row_heights = [1.0 / rows] * rows
    return rows, cols, positions, specs, row_heights


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

    Produces an asymmetric grid (heatmap full-width, profiles|distributions,
    summary full-width) with WebGL rendering, statistical envelopes, and
    cross-panel highlighting support.

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

    if truncate_labels:
        display_names = [
            n[:truncate_labels] + "\u2026" if len(n) > truncate_labels else n
            for n in names
        ]
    else:
        display_names = list(names)

    rows, cols, pos_map, specs, row_heights = _compute_subplot_grid(panels)

    if height is None:
        base = 500 if rows == 1 else 800
        height = min(1400, max(base, 120 + N * 18))

    colours = _generate_palette(N)

    if line_opacity is None:
        line_opacity = max(0.15, 1.0 - N * 0.03)

    ref_aa = []

    # Build subplot titles in grid order (skipping None specs)
    _title_lookup = {
        "heatmap": f"Heatmap \u2014 {metric}",
        "profiles": f"Per-Residue Profiles \u2014 {metric}",
        "distributions": f"Distributions \u2014 {metric}",
        "summary": f"Summary \u2014 {metric}",
    }
    title_by_pos = {pos_map[p]: _title_lookup[p] for p in panels}
    ordered_titles = []
    for r_idx in range(rows):
        for c_idx in range(cols):
            if specs[r_idx][c_idx] is not None:
                ordered_titles.append(title_by_pos.get((r_idx + 1, c_idx + 1), ""))

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=ordered_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        specs=specs,
        row_heights=row_heights,
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
                title=f"{metric} (log)", x=1.02, len=0.4, y=0.95, yanchor="top",
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
                colorbar=dict(title=metric, x=1.02, len=0.4, y=0.95, yanchor="top"),
                customdata=aa_customdata,
                hovertemplate="Residue %{x} (%{customdata})<br>%{y}<br>Value: %{z:.4f}<extra></extra>",
            ), row=r, col=c)

        # Heatmap mutation overlay — keep go.Scatter (categorical y-axis)
        if highlight_mutations and mut_per_protein:
            mut_x, mut_y, mut_text = [], [], []
            for i, name in enumerate(names):
                muts = mut_per_protein.get(name, {})
                for pos, label in muts.items():
                    mut_x.append(pos)
                    mut_y.append(display_names[i])
                    mut_text.append(f"{label[0]}\u2192{label[-1]}")
            if mut_x:
                fig.add_trace(go.Scatter(
                    x=mut_x, y=mut_y, mode="markers",
                    marker=dict(symbol="x", size=7, color="red",
                                line=dict(width=1, color="darkred")),
                    text=mut_text,
                    hovertemplate="Res %{x} (%{text})<br>%{y}<extra>Mutation</extra>",
                    showlegend=False,
                ), row=r, col=c)

    # --- Profiles ---
    if "profiles" in pos_map:
        r, c = pos_map["profiles"]
        if not ref_aa:
            _any_df = next(iter(df_dict.values()))
            _max_res = int(_any_df["residue_index"].max()) if "residue_index" in _any_df.columns else 0
            ref_aa = _extract_ref_sequence(df_dict, _max_res) if _max_res else []

        # Mean + IQR envelope for large N
        if N > 20:
            _any_df = next(iter(df_dict.values()))
            _max_res = int(_any_df["residue_index"].max()) if "residue_index" in _any_df.columns else 0
            if _max_res:
                all_arrays = []
                for name in names:
                    df = df_dict[name]
                    if metric not in df.columns:
                        continue
                    arr = np.full(_max_res, np.nan)
                    idx = df["residue_index"].values.astype(int) - 1
                    vals = df[metric].values.astype(float)
                    valid_idx = (idx >= 0) & (idx < _max_res)
                    arr[idx[valid_idx]] = vals[valid_idx]
                    all_arrays.append(arr)
                if all_arrays:
                    stacked = np.array(all_arrays)
                    with np.errstate(all="ignore"):
                        mean_line = np.nanmean(stacked, axis=0)
                        q25 = np.nanpercentile(stacked, 25, axis=0)
                        q75 = np.nanpercentile(stacked, 75, axis=0)
                    x_range = np.arange(1, _max_res + 1)
                    fig.add_trace(go.Scattergl(
                        x=x_range, y=q75, mode="lines",
                        line=dict(width=0), showlegend=False,
                        hoverinfo="skip",
                        meta={"seqName": "__envelope__"},
                    ), row=r, col=c)
                    fig.add_trace(go.Scattergl(
                        x=x_range, y=q25, mode="lines",
                        line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(100, 100, 100, 0.15)",
                        showlegend=False, hoverinfo="skip",
                        name="IQR (25\u201375%)",
                        meta={"seqName": "__envelope__"},
                    ), row=r, col=c)
                    fig.add_trace(go.Scattergl(
                        x=x_range, y=mean_line, mode="lines",
                        line=dict(color="rgba(0,0,0,0.7)", width=2.5, dash="dot"),
                        name="Mean", showlegend=True,
                        hovertemplate="Res %{x}<br>Mean: %{y:.4f}<extra>Mean</extra>",
                        meta={"seqName": "__mean__"},
                    ), row=r, col=c)

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
            show_in_legend = True if N <= 15 or i < 5 else False
            fig.add_trace(go.Scattergl(
                x=res_idx, y=df[metric].values, mode="lines",
                line=dict(color=colours[i], width=1.2), opacity=line_opacity,
                name=display_names[i], legendgroup=display_names[i],
                showlegend=show_in_legend,
                text=hover_text,
                hovertemplate="%{text}<br>%{y:.4f}<extra>%{fullData.name}</extra>",
                meta={"seqName": names[i]},
            ), row=r, col=c)
        if highlight_mutations and mut_per_protein:
            prof_mut_x, prof_mut_y, prof_mut_text, prof_mut_color = [], [], [], []
            for i, name in enumerate(names):
                df = df_dict[name]
                if metric not in df.columns:
                    continue
                muts = mut_per_protein.get(name, {})
                if not muts:
                    continue
                res_idx = df["residue_index"].values.astype(int)
                vals = df[metric].values
                idx_to_val = dict(zip(res_idx, vals))
                for pos, label in muts.items():
                    if pos in idx_to_val:
                        prof_mut_x.append(pos)
                        prof_mut_y.append(idx_to_val[pos])
                        prof_mut_text.append(f"{display_names[i]}: {label[0]}\u2192{label[-1]}")
                        prof_mut_color.append(colours[i])
            if prof_mut_x:
                fig.add_trace(go.Scattergl(
                    x=prof_mut_x, y=prof_mut_y, mode="markers",
                    marker=dict(symbol="circle", size=6, color=prof_mut_color,
                                line=dict(width=1, color="darkred")),
                    text=prof_mut_text,
                    hovertemplate="Res %{x}<br>%{y:.4f}<br>%{text}<extra>Mutation</extra>",
                    showlegend=False,
                ), row=r, col=c)

    # --- Distributions ---
    if "distributions" in pos_map:
        r, c = pos_map["distributions"]
        if N > 15:
            top_dist_names = names[:15]
            rest_dist_names = names[15:]
            for i, name in enumerate(top_dist_names):
                df = df_dict[name]
                if metric not in df.columns:
                    continue
                vals = df[metric].dropna().values
                if show_violin:
                    fig.add_trace(go.Violin(
                        y=vals, name=display_names[i], legendgroup=display_names[i],
                        showlegend=False, line_color=colours[i],
                        meanline_visible=True, scalemode="width", width=0.8,
                        meta={"seqName": names[i]},
                    ), row=r, col=c)
                else:
                    fig.add_trace(go.Box(
                        y=vals, name=display_names[i], legendgroup=display_names[i],
                        showlegend=False, marker_color=colours[i],
                        meta={"seqName": names[i]},
                    ), row=r, col=c)
            if rest_dist_names:
                rest_vals = []
                for name in rest_dist_names:
                    df = df_dict[name]
                    if metric in df.columns:
                        rest_vals.extend(df[metric].dropna().values.tolist())
                if rest_vals:
                    others_label = f"Others ({len(rest_dist_names)})"
                    if show_violin:
                        fig.add_trace(go.Violin(
                            y=rest_vals, name=others_label,
                            showlegend=False, line_color="gray",
                            meanline_visible=True, scalemode="width", width=0.8,
                            meta={"seqName": "__others__"},
                        ), row=r, col=c)
                    else:
                        fig.add_trace(go.Box(
                            y=rest_vals, name=others_label,
                            showlegend=False, marker_color="gray",
                            meta={"seqName": "__others__"},
                        ), row=r, col=c)
        else:
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
                        meta={"seqName": names[i]},
                    ), row=r, col=c)
                else:
                    fig.add_trace(go.Box(
                        y=vals, name=display_names[i], legendgroup=display_names[i],
                        showlegend=False, marker_color=colours[i],
                        meta={"seqName": names[i]},
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
            meta={"seqName": "__summary__"},
        ), row=r, col=c)
        fig.add_trace(go.Scattergl(
            x=display_names, y=max_vals, mode="markers",
            marker=dict(symbol="diamond", size=9, color=colours,
                        line=dict(width=1, color="black")),
            name="Max", showlegend=False,
            hovertemplate="%{x}<br>Max: %{y:.4f}<extra></extra>",
            meta={"seqName": "__summary__"},
        ), row=r, col=c)

    # --- Layout polish ---
    y_tick_size = max(7, min(11, 200 // max(N, 1)))
    for panel, (r, c) in pos_map.items():
        if panel in ("heatmap", "profiles"):
            fig.update_xaxes(title_text="Residue Index", row=r, col=c)
        if panel == "heatmap":
            fig.update_yaxes(tickfont=dict(size=y_tick_size), row=r, col=c)
        if panel in ("profiles", "distributions"):
            fig.update_yaxes(title_text=metric, row=r, col=c,
                             type="log" if log_y else "linear")
        if panel == "summary":
            fig.update_xaxes(title_text="Protein", tickangle=45, row=r, col=c)
            fig.update_yaxes(title_text=metric, row=r, col=c,
                             type="log" if log_y else "linear")

    dashboard_title = title or f"Multi-Protein Dashboard \u2014 {metric}"
    if N > 15:
        legend_cfg = dict(
            orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5,
            font=dict(size=10),
        )
    else:
        legend_cfg = dict(
            orientation="v", yanchor="top", y=0.45, xanchor="left", x=1.02,
        )
    fig.update_layout(
        title=dict(text=dashboard_title, x=0, xanchor="left"),
        height=height, width=width,
        autosize=True, template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
        legend=legend_cfg,
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

    Features lazy tab loading, cross-panel highlighting, modern CSS theme,
    and enhanced search/filter controls.

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
        safe_id = f"tab-{i}"
        if i == 0:
            div_html = fig.to_html(include_plotlyjs=True, full_html=False)
            tab_divs.append(
                f'<div id="{safe_id}" class="tab-content" style="display:block;">'
                f'{div_html}</div>'
            )
        else:
            fig_json = fig.to_json()
            tab_divs.append(
                f'<div id="{safe_id}" class="tab-content" style="display:none;">'
                f'<div class="plot-placeholder" id="plot-{safe_id}"></div>'
                f'<script type="application/json" id="data-{safe_id}">'
                f'{fig_json}</script></div>'
            )

    buttons = []
    for i, name in enumerate(tab_names):
        active = ' class="active"' if i == 0 else ""
        buttons.append(f'<button{active} onclick="switchTab({i})">{name}</button>')
    button_bar = "".join(buttons)

    css = """
<style>
:root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8f9fb;
    --bg-hover: #edf0f5;
    --border-color: #e2e5ea;
    --text-primary: #1a1d23;
    --text-secondary: #5f6778;
    --accent: #4f6ef7;
    --accent-light: #eef1fe;
    --radius: 8px;
    --font: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
*, *::before, *::after { box-sizing: border-box; }
html, body { height: 100%; margin: 0; padding: 0; font-family: var(--font);
    background: var(--bg-secondary); color: var(--text-primary); }
body { display: flex; flex-direction: column; }
.dash-header {
    background: var(--bg-primary); border-bottom: 1px solid var(--border-color);
    padding: 16px 24px 0; flex-shrink: 0;
}
.dash-header h1 {
    margin: 0 0 12px; font-size: 20px; font-weight: 600;
    color: var(--text-primary); letter-spacing: -0.3px;
}
.tab-bar { display: flex; gap: 4px; margin: 0; }
.tab-bar button {
    padding: 8px 20px; border: 1px solid var(--border-color); border-bottom: none;
    border-radius: var(--radius) var(--radius) 0 0;
    background: var(--bg-secondary); cursor: pointer;
    font-size: 13px; font-weight: 500; color: var(--text-secondary);
    font-family: var(--font); transition: all 0.15s ease;
}
.tab-bar button:hover { background: var(--bg-hover); color: var(--text-primary); }
.tab-bar button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.tab-bar button.active {
    background: var(--bg-primary); border-bottom: 2px solid var(--bg-primary);
    margin-bottom: -1px; color: var(--accent); font-weight: 600;
}
.control-bar {
    background: var(--bg-primary); padding: 10px 24px;
    border-bottom: 1px solid var(--border-color);
    display: flex; gap: 12px; align-items: center; flex-shrink: 0; flex-wrap: wrap;
}
.control-bar label {
    font-size: 12px; color: var(--text-secondary);
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
}
.search-wrapper {
    position: relative; display: inline-flex; align-items: center;
}
.search-wrapper input[type="text"] {
    padding: 6px 28px 6px 10px; border: 1px solid var(--border-color);
    border-radius: var(--radius); font-size: 13px; width: 220px;
    font-family: var(--font); background: var(--bg-secondary);
    transition: border-color 0.15s, box-shadow 0.15s;
}
.search-wrapper input[type="text"]:focus {
    outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-light);
}
.search-clear {
    position: absolute; right: 6px; top: 50%; transform: translateY(-50%);
    background: none; border: none; cursor: pointer; color: var(--text-secondary);
    font-size: 16px; padding: 2px; line-height: 1; display: none;
}
.search-clear:hover { color: var(--text-primary); }
.control-bar select {
    padding: 6px 10px; border: 1px solid var(--border-color);
    border-radius: var(--radius); font-size: 13px;
    font-family: var(--font); background: var(--bg-secondary); cursor: pointer;
}
.control-bar select:focus {
    outline: none; border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-light);
}
.result-count { font-size: 12px; color: var(--text-secondary); margin-left: 4px; }
.kbd-hint {
    font-size: 11px; color: var(--text-secondary); margin-left: auto; opacity: 0.7;
}
.kbd-hint kbd {
    background: var(--bg-secondary); border: 1px solid var(--border-color);
    border-radius: 3px; padding: 1px 5px; font-size: 10px; font-family: var(--font);
}
.tab-content { flex: 1; padding: 8px 16px; overflow: auto; background: var(--bg-secondary); }
.tab-content .plotly-graph-div { width: 100% !important; }
.plot-placeholder {
    width: 100%; min-height: 400px; display: flex;
    align-items: center; justify-content: center; color: var(--text-secondary);
}
</style>
"""

    js = """
<script>
/* --- Tab switching with lazy loading --- */
var _tabInitialized = {0: true};

function switchTab(idx) {
    var tabs = document.querySelectorAll('.tab-content');
    var buttons = document.querySelectorAll('.tab-bar button');
    for (var i = 0; i < tabs.length; i++) {
        tabs[i].style.display = (i === idx) ? 'block' : 'none';
        buttons[i].className = (i === idx) ? 'active' : '';
    }
    if (!_tabInitialized[idx]) {
        _tabInitialized[idx] = true;
        var tab = tabs[idx];
        var jsonScript = tab.querySelector('script[type="application/json"]');
        var placeholder = tab.querySelector('.plot-placeholder');
        if (jsonScript && placeholder) {
            var figData = JSON.parse(jsonScript.textContent);
            Plotly.newPlot(placeholder, figData.data, figData.layout, {responsive: true});
            _setupHighlighting(placeholder);
        }
    }
    var visibleTab = tabs[idx];
    var plots = visibleTab.querySelectorAll('.plotly-graph-div');
    for (var j = 0; j < plots.length; j++) {
        if (window.Plotly) { Plotly.Plots.resize(plots[j]); }
    }
    _origData = {};
    applyFilter();
}

window.addEventListener('resize', function() {
    var plots = document.querySelectorAll('.tab-content[style*="block"] .plotly-graph-div');
    for (var i = 0; i < plots.length; i++) {
        if (window.Plotly) { Plotly.Plots.resize(plots[i]); }
    }
});

/* --- Cross-panel highlighting --- */
var _highlightedSeq = null;
var _origStyles = {};

function _setupHighlighting(plotEl) {
    plotEl.on('plotly_click', function(data) {
        if (!data || !data.points || !data.points.length) return;
        var pt = data.points[0];
        var meta = pt.data.meta;
        var seqName = meta ? meta.seqName : null;
        if (!seqName || seqName.indexOf('__') === 0) return;
        if (_highlightedSeq === seqName) {
            _highlightedSeq = null;
            _resetHighlight();
        } else {
            _highlightedSeq = seqName;
            _applyHighlight(seqName);
        }
    });
}

function _ensureOrigStyles(plot) {
    if (_origStyles[plot.id]) return;
    _origStyles[plot.id] = plot.data.map(function(t) {
        return {
            opacity: t.opacity,
            lineWidth: (t.line && t.line.width) ? t.line.width : undefined
        };
    });
}

function _applyHighlight(seqName) {
    var plots = document.querySelectorAll('.tab-content[style*="block"] .plotly-graph-div');
    plots.forEach(function(plot) {
        if (!plot.data) return;
        _ensureOrigStyles(plot);
        for (var i = 0; i < plot.data.length; i++) {
            var trace = plot.data[i];
            var meta = trace.meta;
            var tSeq = meta ? meta.seqName : null;
            if (!tSeq || tSeq.indexOf('__') === 0) continue;
            if (tSeq === seqName) {
                Plotly.restyle(plot, {opacity: 1, 'line.width': 3}, [i]);
            } else {
                Plotly.restyle(plot, {opacity: 0.08}, [i]);
            }
        }
    });
}

function _resetHighlight() {
    var plots = document.querySelectorAll('.tab-content[style*="block"] .plotly-graph-div');
    plots.forEach(function(plot) {
        if (!plot.data) return;
        var orig = _origStyles[plot.id];
        if (!orig) return;
        for (var i = 0; i < plot.data.length; i++) {
            var meta = plot.data[i].meta;
            if (!meta || !meta.seqName || meta.seqName.indexOf('__') === 0) continue;
            var o = orig[i] || {};
            var update = {};
            update.opacity = o.opacity !== undefined ? o.opacity : null;
            if (o.lineWidth !== undefined) {
                update['line.width'] = o.lineWidth;
            }
            Plotly.restyle(plot, update, [i]);
        }
    });
    _origStyles = {};
}

/* --- Search & Top-N filtering --- */
var _origData = {};
var _totalCount = 0;

function _getActivePlots() {
    var tab = document.querySelector('.tab-content[style*="display: block"], .tab-content[style*="display:block"]');
    if (!tab) return [];
    return Array.from(tab.querySelectorAll('.plotly-graph-div'));
}

function _initOrigData(plots) {
    plots.forEach(function(plot) {
        if (_origData[plot.id]) return;
        var data = plot.data || [];
        var names = data.map(function(t) { return t.name || ''; });
        var entry = { traceNames: names };
        for (var i = 0; i < data.length; i++) {
            if (data[i].type === 'heatmap') {
                entry.heatmapIdx = i;
                entry.heatmapZ = JSON.parse(JSON.stringify(data[i].z));
                entry.heatmapY = data[i].y.slice();
                if (data[i].customdata) {
                    entry.heatmapCD = JSON.parse(JSON.stringify(data[i].customdata));
                }
                if (!_totalCount) _totalCount = entry.heatmapY.length;
                break;
            }
        }
        if (!_totalCount) {
            var uniqueNames = new Set(names.filter(function(n) {
                return n && n !== 'Mean' && n !== 'Max' && n.indexOf('IQR') !== 0;
            }));
            _totalCount = uniqueNames.size;
        }
        _origData[plot.id] = entry;
    });
}

function _updateResultCount(shown) {
    var el = document.getElementById('result-count');
    if (el) el.textContent = 'Showing ' + shown + ' of ' + (_totalCount || '?');
}

function applyFilter() {
    var query = (document.getElementById('search-input').value || '').toLowerCase();
    var topN = document.getElementById('topn-select').value;
    var plots = _getActivePlots();
    _initOrigData(plots);

    var shownCount = 0;

    plots.forEach(function(plot) {
        var orig = _origData[plot.id];
        if (!orig) return;
        var data = plot.data || [];

        if (orig.heatmapIdx !== undefined) {
            var hIdx = orig.heatmapIdx;
            var matchIdx = [];
            for (var r = 0; r < orig.heatmapY.length; r++) {
                if (!query || orig.heatmapY[r].toLowerCase().indexOf(query) !== -1) {
                    matchIdx.push(r);
                }
            }
            if (topN !== 'all') {
                matchIdx = matchIdx.slice(0, parseInt(topN));
            }
            shownCount = Math.max(shownCount, matchIdx.length);
            var newZ = matchIdx.map(function(i) { return orig.heatmapZ[i]; });
            var newY = matchIdx.map(function(i) { return orig.heatmapY[i]; });
            var update = { z: [newZ], y: [newY] };
            if (orig.heatmapCD) {
                update.customdata = [matchIdx.map(function(i) { return orig.heatmapCD[i]; })];
            }
            Plotly.restyle(plot, update, [hIdx]);

            for (var s = 0; s < data.length; s++) {
                if (s === hIdx) continue;
                if (data[s].type === 'scatter' && data[s].mode === 'markers') {
                    var oy = (plot.data[s]._origY || data[s].y);
                    if (!plot.data[s]._origY) {
                        plot.data[s]._origY = data[s].y.slice();
                        plot.data[s]._origX = data[s].x.slice();
                        plot.data[s]._origText = data[s].text ? data[s].text.slice() : null;
                    }
                    var ySet = new Set(newY);
                    var fX = [], fY = [], fT = [];
                    for (var k = 0; k < plot.data[s]._origY.length; k++) {
                        if (ySet.has(plot.data[s]._origY[k])) {
                            fX.push(plot.data[s]._origX[k]);
                            fY.push(plot.data[s]._origY[k]);
                            if (plot.data[s]._origText) fT.push(plot.data[s]._origText[k]);
                        }
                    }
                    var sUpdate = { x: [fX], y: [fY] };
                    if (plot.data[s]._origText) sUpdate.text = [fT];
                    Plotly.restyle(plot, sUpdate, [s]);
                }
            }
        } else {
            var matchNames = new Set();
            var count = 0;
            var limit = (topN !== 'all') ? parseInt(topN) : Infinity;
            orig.traceNames.forEach(function(n) {
                if (n && (!query || n.toLowerCase().indexOf(query) !== -1) && count < limit) {
                    matchNames.add(n);
                    count++;
                }
            });
            shownCount = Math.max(shownCount, matchNames.size);
            for (var t = 0; t < data.length; t++) {
                var tname = orig.traceNames[t] || '';
                var keep = !tname || tname === 'Mean' || tname === 'Max'
                    || tname.indexOf('IQR') === 0 || matchNames.has(tname);
                Plotly.restyle(plot, { visible: keep ? true : 'legendonly' }, [t]);
            }
        }
    });
    _updateResultCount(shownCount);
    var clearBtn = document.getElementById('search-clear');
    if (clearBtn) clearBtn.style.display = query ? 'block' : 'none';
}

function clearSearch() {
    var input = document.getElementById('search-input');
    if (input) { input.value = ''; }
    applyFilter();
    if (input) input.focus();
}

document.addEventListener('DOMContentLoaded', function() {
    var search = document.getElementById('search-input');
    var topn = document.getElementById('topn-select');
    if (search) {
        var timer;
        search.addEventListener('input', function() {
            clearTimeout(timer);
            timer = setTimeout(applyFilter, 250);
        });
    }
    if (topn) topn.addEventListener('change', applyFilter);

    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
            if (search) {
                e.preventDefault();
                search.focus();
                search.select();
            }
        }
    });

    var plots = document.querySelectorAll('.plotly-graph-div');
    plots.forEach(function(p) { _setupHighlighting(p); });
});
</script>
"""

    control_bar = (
        '<div class="control-bar">'
        '<label for="search-input">Search</label>'
        '<div class="search-wrapper">'
        '<input type="text" id="search-input" placeholder="Filter by name\u2026">'
        '<button class="search-clear" id="search-clear" onclick="clearSearch()" '
        'title="Clear">&times;</button>'
        '</div>'
        '<label for="topn-select">Show</label>'
        '<select id="topn-select">'
        '<option value="10">Top 10</option>'
        '<option value="20" selected>Top 20</option>'
        '<option value="50">Top 50</option>'
        '<option value="100">Top 100</option>'
        '<option value="all">All</option>'
        '</select>'
        '<span class="result-count" id="result-count"></span>'
        '<span class="kbd-hint"><kbd>\u2318</kbd>+<kbd>F</kbd> to search</span>'
        '</div>'
    )

    html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{page_title}</title>{css}</head><body>"
        f'<div class="dash-header"><h1>{page_title}</h1>'
        f'<div class="tab-bar">{button_bar}</div></div>'
        f"{control_bar}{''.join(tab_divs)}{js}</body></html>"
    )
    output.write_text(html)
    return str(output)
