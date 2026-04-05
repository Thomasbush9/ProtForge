#!/usr/bin/env python3
"""Correlation analysis plots for ProtForge ES metrics vs phenotype data.

Produces a multi-tab HTML report with:
    1. Per-Residue Correlation -- Pearson r at each residue position
    2. Metric vs Phenotype     -- scatter plots of aggregated metric vs phenotype
    3. Summary                 -- horizontal bar chart of overall correlations

Usage:
    python -m viz.correlation \
        --es_dir /path/to/es \
        --phenotype /path/to/phenotype.tsv \
        --phenotype_col medianBrightness \
        --metric strain \
        --html correlation.html
"""

import argparse
import sys

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


def _get_mutation_positions(es_dir):
    """Scan ES CSVs and return residue indices where mutations occur.

    Returns a sorted list of unique residue positions where at least one
    sequence has ``protA_resname != protB_resname``.
    """
    from pathlib import Path
    mut_positions = set()
    es_path = Path(es_dir)
    for csv_file in es_path.glob("*.csv"):
        if csv_file.name in ("combined.csv",):
            continue
        try:
            df = pd.read_csv(csv_file, usecols=["residue_index", "protA_resname", "protB_resname"])
            muts = df[df["protA_resname"] != df["protB_resname"]]
            mut_positions.update(muts["residue_index"].astype(int).tolist())
        except Exception:
            continue
    return sorted(mut_positions)


def plot_per_residue_correlation(per_residue_df, metric="strain", phenotype_col="medianBrightness",
                                 mutation_positions=None):
    """Publication-style line plot of Pearson r vs residue position.

    Thick black line with dotted verticals at mutation sites and a red
    circle highlighting the residue with the strongest correlation.

    Parameters
    ----------
    per_residue_df : DataFrame
        Output from ``correlate_metric_with_phenotype()["per_residue"]``.
        Columns: ``residue_index``, ``correlation``, ``p_value``, ``n_sequences``.
    metric : str
        Metric name (for axis labels).
    phenotype_col : str
        Phenotype name (for axis labels).
    mutation_positions : list[int] or None
        Residue indices where mutations occur (shown as dotted verticals).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if per_residue_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No per-residue correlation data available",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    df = per_residue_df.sort_values("residue_index").reset_index(drop=True)
    has_spearman = "spearman" in df.columns

    # Short labels for axes
    metric_short = "S" if metric == "strain" else metric[0].upper()
    pheno_short = "Fluor." if "bright" in phenotype_col.lower() or "fluor" in phenotype_col.lower() else phenotype_col

    fig = go.Figure()

    # Spearman line (behind Pearson) — blue, thinner
    if has_spearman:
        fig.add_trace(go.Scatter(
            x=df["residue_index"],
            y=df["spearman"],
            mode="lines",
            line=dict(color="rgba(99,110,250,0.6)", width=1.5),
            name=f"Spearman \u03c1",
            customdata=np.column_stack([df["spearman_p_value"], df["n_sequences"]]),
            hovertemplate=(
                "Residue %{x}<br>"
                "\u03c1 = %{y:.3f}<br>"
                "p = %{customdata[0]:.2e}<br>"
                "n = %{customdata[1]}<br>"
                "<extra></extra>"
            ),
        ))

    # Main Pearson line — thick black, paper-style
    hover_custom = [df["p_value"], df["n_sequences"]]
    hover_tmpl = ("Residue %{x}<br>"
                  "r = %{y:.3f}<br>"
                  "p = %{customdata[0]:.2e}<br>")
    if has_spearman:
        hover_custom.append(df["spearman"])
        hover_tmpl += "\u03c1 = %{customdata[2]:.3f}<br>"
    hover_tmpl += "n = %{customdata[1]}<br><extra></extra>"

    fig.add_trace(go.Scatter(
        x=df["residue_index"],
        y=df["correlation"],
        mode="lines",
        line=dict(color="black", width=2),
        name=f"Pearson r",
        customdata=np.column_stack(hover_custom),
        hovertemplate=hover_tmpl,
    ))

    # Highlight the strongest Spearman correlation with a red open circle
    corr_col = "spearman" if has_spearman else "correlation"
    min_idx = df[corr_col].abs().idxmax()
    min_row = df.loc[min_idx]
    label_r = f"\u03c1={min_row['spearman']:.2f}" if has_spearman else f"r={min_row['correlation']:.2f}"
    fig.add_trace(go.Scatter(
        x=[min_row["residue_index"]],
        y=[min_row[corr_col]],
        mode="markers+text",
        marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=2, color="red")),
        text=[f"Res {int(min_row['residue_index'])} ({label_r})"],
        textposition="bottom center",
        textfont=dict(size=10, color="red"),
        showlegend=False,
    ))

    # Dotted verticals at mutation positions
    if mutation_positions:
        y_min = df["correlation"].min()
        for pos in mutation_positions:
            fig.add_shape(
                type="line",
                x0=pos, x1=pos,
                y0=y_min - 0.05, y1=0.05,
                line=dict(color="grey", width=1, dash="dot"),
                layer="below",
            )

    y_min = df["correlation"].min()
    if has_spearman:
        y_min = min(y_min, df["spearman"].min())

    fig.update_layout(
        xaxis_title="Sequence position",
        yaxis_title=f"Corr ({metric_short}<sub>n</sub>, {pheno_short})",
        yaxis=dict(range=[y_min - 0.1, 0.15]),
        xaxis=dict(dtick=50),
        template="plotly_white",
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        height=400,
        margin=dict(t=30, b=60, l=70, r=30),
        plot_bgcolor="white",
    )

    return fig


def plot_single_residue_scatter(es_dir, phenotype_path, residue_index, metric="strain",
                                phenotype_col="medianBrightness", phenotype_sep="\t",
                                log_x=True, n_bins=20):
    """Scatter plot of a single residue's metric vs phenotype with smoothed median.

    Paper-style: green scatter, black median curve, log-x axis,
    r annotation, axis labels S_{L<residue>}.

    Parameters
    ----------
    es_dir : str
        ES output directory.
    phenotype_path : str
        Phenotype file path.
    residue_index : int
        Residue position to plot.
    metric : str
        Metric column name.
    phenotype_col : str
        Phenotype column name.
    phenotype_sep : str
        Separator for phenotype file.
    log_x : bool
        Use log scale on x-axis.
    n_bins : int
        Number of bins for the rolling median curve.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    from scipy import stats as sp_stats
    from utils.utils import load_es_data

    pheno_df = pd.read_csv(phenotype_path, sep=phenotype_sep)
    es_data = load_es_data(es_dir, metric=metric)

    records = []
    for seq_name, df in es_data.items():
        idx_str = seq_name.replace("seq_", "")
        try:
            seq_idx = int(idx_str)
        except ValueError:
            continue
        if seq_idx >= len(pheno_df):
            continue
        phenotype_val = pheno_df.iloc[seq_idx].get(phenotype_col)
        if pd.isna(phenotype_val):
            continue
        row = df[df["residue_index"] == residue_index]
        if row.empty or pd.isna(row[metric].values[0]):
            continue
        records.append({
            "metric_val": float(row[metric].values[0]),
            phenotype_col: float(phenotype_val),
            "seq_name": seq_name,
        })

    if not records:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for residue {residue_index}",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    data = pd.DataFrame(records)
    x = data["metric_val"].values
    y = data[phenotype_col].values

    # Correlations
    x_for_corr = np.log10(x + 1e-10) if log_x else x
    valid = np.isfinite(x_for_corr) & np.isfinite(y)
    r_pearson, _ = sp_stats.pearsonr(x_for_corr[valid], y[valid])
    r_spearman, _ = sp_stats.spearmanr(x[valid], y[valid])

    # Short labels
    pheno_short = "Fluor." if "bright" in phenotype_col.lower() or "fluor" in phenotype_col.lower() else phenotype_col
    metric_short = "S" if metric == "strain" else metric[0].upper()

    fig = go.Figure()

    # Scatter — green dots
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(size=4, color="rgba(100,200,100,0.5)", line=dict(width=0)),
        text=data["seq_name"],
        hovertemplate="%{text}<br>" + f"{metric_short}" + "<sub>%{x:.4f}</sub><br>" + f"{pheno_short}" + " = %{y:.3f}<extra></extra>",
        showlegend=False,
    ))

    # Smoothed median curve — sort by x, bin, compute median per bin
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    if log_x:
        bin_edges = np.logspace(np.log10(max(x_sorted.min(), 1e-10)),
                                np.log10(x_sorted.max()), n_bins + 1)
    else:
        bin_edges = np.linspace(x_sorted.min(), x_sorted.max(), n_bins + 1)

    bin_x, bin_y = [], []
    for i in range(len(bin_edges) - 1):
        mask = (x_sorted >= bin_edges[i]) & (x_sorted < bin_edges[i + 1])
        if mask.sum() >= 3:
            bin_x.append(np.median(x_sorted[mask]))
            bin_y.append(np.median(y_sorted[mask]))

    if len(bin_x) >= 3:
        fig.add_trace(go.Scatter(
            x=bin_x, y=bin_y,
            mode="lines",
            line=dict(color="black", width=3),
            name="median",
            showlegend=False,
            hoverinfo="skip",
        ))

    # Correlation annotations
    fig.add_annotation(
        text=f"<i>r</i> = {r_pearson:.2f}<br><i>\u03c1</i> = {r_spearman:.2f}",
        xref="paper", yref="paper",
        x=0.95, y=0.15,
        showarrow=False,
        font=dict(size=13),
        align="right",
    )

    axis_type = "log" if log_x else "linear"
    fig.update_layout(
        xaxis_title=f"{metric_short}<sub>L{residue_index}</sub>",
        yaxis_title=pheno_short,
        xaxis=dict(type=axis_type),
        template="plotly_white",
        height=450,
        width=500,
        margin=dict(t=30, b=60, l=70, r=30),
    )

    return fig


def plot_per_sequence_scatter(per_sequence_df, metric="strain", phenotype_col="medianBrightness",
                              aggregations=None):
    """Scatter plots of aggregated metric vs phenotype with OLS trendlines.

    Parameters
    ----------
    per_sequence_df : DataFrame
        Output from ``correlate_metric_with_phenotype()["per_sequence"]``.
    metric : str
        Metric name.
    phenotype_col : str
        Phenotype column name.
    aggregations : list[str] or None
        Aggregations to plot (default: ``["mean", "max", "sum"]``).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if aggregations is None:
        aggregations = ["mean", "max", "sum"]

    # Filter to aggregations that exist in the data
    agg_cols = {agg: f"{metric}_{agg}" for agg in aggregations
                if f"{metric}_{agg}" in per_sequence_df.columns}

    if not agg_cols:
        fig = go.Figure()
        fig.add_annotation(text="No aggregation data available",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    n_plots = len(agg_cols)
    fig = make_subplots(
        rows=1, cols=n_plots,
        subplot_titles=[f"{agg.capitalize()} {metric}" for agg in agg_cols],
        horizontal_spacing=0.08,
    )

    from scipy import stats as sp_stats

    for i, (agg, col) in enumerate(agg_cols.items(), 1):
        valid = per_sequence_df[[col, phenotype_col]].dropna()
        x = valid[col].values
        y = valid[phenotype_col].values

        # Scatter
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(size=5, opacity=0.6, color="#636EFA"),
            name=agg,
            text=per_sequence_df.loc[valid.index, "seq_name"] if "seq_name" in per_sequence_df.columns else None,
            hovertemplate=(
                "%{text}<br>" if "seq_name" in per_sequence_df.columns else ""
            ) + (
                f"{agg} {metric}" + " = %{x:.4f}<br>"
                f"{phenotype_col}" + " = %{y:.4f}<br>"
                "<extra></extra>"
            ),
            showlegend=False,
        ), row=1, col=i)

        # OLS trendline
        if len(x) >= 3:
            r, p = sp_stats.pearsonr(x, y)
            rho, p_s = sp_stats.spearmanr(x, y)
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept

            fig.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ), row=1, col=i)

            # Annotation with r, rho, p
            stars_r = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            stars_s = "***" if p_s < 0.001 else "**" if p_s < 0.01 else "*" if p_s < 0.05 else "ns"
            fig.add_annotation(
                text=f"r = {r:.3f} ({stars_r})<br>\u03c1 = {rho:.3f} ({stars_s})",
                xref=f"x{i}" if i > 1 else "x",
                yref=f"y{i}" if i > 1 else "y",
                x=x.min() + (x.max() - x.min()) * 0.05,
                y=y.max() - (y.max() - y.min()) * 0.05,
                showarrow=False,
                font=dict(size=11),
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="grey",
                borderwidth=1,
            )

        fig.update_xaxes(title_text=f"{agg} {metric}", row=1, col=i)
        if i == 1:
            fig.update_yaxes(title_text=phenotype_col, row=1, col=i)

    fig.update_layout(
        title=f"Aggregated {metric} vs {phenotype_col}",
        template="plotly_white",
        height=450,
        width=max(400 * n_plots, 800),
    )

    return fig


def plot_overall_summary(overall_df, metric="strain", phenotype_col="medianBrightness"):
    """Horizontal bar chart summarising overall correlations per aggregation.

    Parameters
    ----------
    overall_df : DataFrame
        Output from ``correlate_metric_with_phenotype()["overall"]``.
        Columns: ``aggregation``, ``correlation``, ``p_value``, ``n``.
    metric : str
        Metric name (for title).
    phenotype_col : str
        Phenotype name (for title).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if overall_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No overall correlation data available",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    df = overall_df.copy()

    # Significance stars
    def _stars(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    has_spearman = "spearman" in df.columns

    df["stars"] = df["p_value"].apply(_stars)
    df["abs_r"] = df["correlation"].abs()
    df["label_r"] = df.apply(
        lambda row: f"r = {row['correlation']:.3f} {row['stars']}", axis=1
    )

    fig = go.Figure()

    # Pearson bars
    fig.add_trace(go.Bar(
        y=df["aggregation"],
        x=df["correlation"],
        orientation="h",
        marker_color=df["correlation"].apply(lambda r: "#EF553B" if r < 0 else "#636EFA"),
        text=df["label_r"],
        textposition="outside",
        name="Pearson r",
        customdata=np.column_stack([df["p_value"], df["n"]]),
        hovertemplate=(
            "%{y}<br>"
            "r = %{x:.3f}<br>"
            "p = %{customdata[0]:.2e}<br>"
            "n = %{customdata[1]}<br>"
            "<extra></extra>"
        ),
    ))

    # Spearman bars
    if has_spearman:
        df["stars_s"] = df["spearman_p_value"].apply(_stars)
        df["label_s"] = df.apply(
            lambda row: f"\u03c1 = {row['spearman']:.3f} {row['stars_s']}", axis=1
        )
        fig.add_trace(go.Bar(
            y=df["aggregation"],
            x=df["spearman"],
            orientation="h",
            marker_color=df["spearman"].apply(lambda r: "rgba(239,85,59,0.5)" if r < 0 else "rgba(99,110,250,0.5)"),
            text=df["label_s"],
            textposition="outside",
            name="Spearman \u03c1",
            customdata=np.column_stack([df["spearman_p_value"], df["n"]]),
            hovertemplate=(
                "%{y}<br>"
                "\u03c1 = %{x:.3f}<br>"
                "p = %{customdata[0]:.2e}<br>"
                "n = %{customdata[1]}<br>"
                "<extra></extra>"
            ),
        ))

    # Zero line
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)

    x_max = max(df["abs_r"].max() * 1.4, 0.2)
    if has_spearman:
        x_max = max(x_max, df["spearman"].abs().max() * 1.4)
    fig.update_layout(
        title=f"Overall Correlation Summary: {metric} vs {phenotype_col}",
        xaxis_title="Correlation coefficient",
        yaxis_title="Aggregation",
        barmode="group",
        xaxis=dict(range=[-x_max, x_max]),
        template="plotly_white",
        height=300 + len(df) * 40,
    )

    return fig


def build_correlation_report(
    es_dir,
    phenotype_path,
    phenotype_col="medianBrightness",
    metric="strain",
    phenotype_sep="\t",
    aggregations=None,
    output_html="correlation.html",
    title="Correlation Analysis",
    top_residues=3,
):
    """Build a multi-tab correlation report HTML.

    Parameters
    ----------
    es_dir : str
        ES output directory with per-sequence CSV files.
    phenotype_path : str
        Path to phenotype TSV/CSV.
    phenotype_col : str
        Phenotype column name.
    metric : str
        Per-residue metric column.
    phenotype_sep : str
        Separator for phenotype file.
    aggregations : list[str] or None
        Aggregation methods (default: mean, max, sum).
    output_html : str
        Output HTML path.
    title : str
        Page title.
    top_residues : int
        Number of top-correlated residues to generate scatter plots for.

    Returns
    -------
    str
        Path to the written HTML file.
    """
    from utils.utils import correlate_metric_with_phenotype
    from .visualization import write_dashboard_html

    if aggregations is None:
        aggregations = ["mean", "max", "sum"]

    print(f"Computing correlations ({metric} vs {phenotype_col}) ...")
    result = correlate_metric_with_phenotype(
        es_dir=es_dir,
        phenotype_path=phenotype_path,
        phenotype_col=phenotype_col,
        metric=metric,
        phenotype_sep=phenotype_sep,
        aggregations=aggregations,
    )

    per_residue_df = result["per_residue"]
    per_sequence_df = result["per_sequence"]
    overall_df = result["overall"]

    n_seq = len(per_sequence_df)
    n_res = len(per_residue_df)
    print(f"  {n_seq} sequences, {n_res} residues with correlation data.")

    figures = {}

    # Discover mutation positions from ES data
    mut_positions = _get_mutation_positions(es_dir)
    if mut_positions:
        print(f"  {len(mut_positions)} mutation positions detected.")

    # Tab 1: Per-residue correlation
    fig1 = plot_per_residue_correlation(per_residue_df, metric=metric, phenotype_col=phenotype_col,
                                        mutation_positions=mut_positions)
    figures["Per-Residue Correlation"] = fig1

    # Tab 2: Scatter plots
    fig2 = plot_per_sequence_scatter(per_sequence_df, metric=metric, phenotype_col=phenotype_col,
                                     aggregations=aggregations)
    figures["Metric vs Phenotype"] = fig2

    # Tab 3: Summary
    fig3 = plot_overall_summary(overall_df, metric=metric, phenotype_col=phenotype_col)
    figures["Summary"] = fig3

    # Tabs 4+: Single-residue scatter plots for top correlated residues
    if top_residues and not per_residue_df.empty:
        # Rank by Spearman if available, else Pearson
        rank_col = "spearman" if "spearman" in per_residue_df.columns else "correlation"
        top_res = per_residue_df.reindex(
            per_residue_df[rank_col].abs().sort_values(ascending=False).index
        ).head(top_residues)
        metric_short = "S" if metric == "strain" else metric[0].upper()
        for _, row in top_res.iterrows():
            res_idx = int(row["residue_index"])
            r_val = row["correlation"]
            rho_val = row.get("spearman", None)
            corr_str = f"\u03c1={rho_val:.2f}" if rho_val is not None else f"r={r_val:.2f}"
            print(f"  Building scatter for residue {res_idx} (r={r_val:.3f}, \u03c1={rho_val:.3f}) ..." if rho_val is not None else f"  Building scatter for residue {res_idx} (r={r_val:.3f}) ...")
            fig_res = plot_single_residue_scatter(
                es_dir, phenotype_path, res_idx,
                metric=metric, phenotype_col=phenotype_col,
                phenotype_sep=phenotype_sep,
            )
            figures[f"{metric_short}<sub>{res_idx}</sub>  ({corr_str})"] = fig_res

    out = write_dashboard_html(figures, output_html, title=title)
    print(f"\nCorrelation report written to: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build correlation analysis plots (ES metric vs phenotype)."
    )
    parser.add_argument(
        "--es_dir", required=True,
        help="ES output directory with per-sequence CSV files",
    )
    parser.add_argument(
        "--phenotype", required=True,
        help="Path to phenotype TSV/CSV file",
    )
    parser.add_argument(
        "--phenotype_col", default="medianBrightness",
        help="Phenotype column name (default: medianBrightness)",
    )
    parser.add_argument(
        "--phenotype_sep", default="\t",
        help="Phenotype file separator (default: tab)",
    )
    parser.add_argument(
        "--metric", default="strain",
        help="Per-residue metric column (default: strain)",
    )
    parser.add_argument(
        "--aggregations", nargs="+", default=["mean", "max", "sum"],
        help="Aggregation methods (default: mean max sum)",
    )
    parser.add_argument(
        "--html", default="correlation.html",
        help="Output HTML file path (default: correlation.html)",
    )
    parser.add_argument(
        "--title", default="Correlation Analysis",
        help="HTML page title",
    )
    parser.add_argument(
        "--top_residues", type=int, default=3,
        help="Number of top-correlated residues to generate scatter plots for (default: 3)",
    )

    args = parser.parse_args()

    build_correlation_report(
        es_dir=args.es_dir,
        phenotype_path=args.phenotype,
        phenotype_col=args.phenotype_col,
        metric=args.metric,
        phenotype_sep=args.phenotype_sep,
        aggregations=args.aggregations,
        output_html=args.html,
        title=args.title,
        top_residues=args.top_residues,
    )


if __name__ == "__main__":
    main()
