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


def plot_per_residue_correlation(per_residue_df, metric="strain", phenotype_col="medianBrightness"):
    """Line plot of Pearson r vs residue index with significance highlighting.

    Parameters
    ----------
    per_residue_df : DataFrame
        Output from ``correlate_metric_with_phenotype()["per_residue"]``.
        Columns: ``residue_index``, ``correlation``, ``p_value``, ``n_sequences``.
    metric : str
        Metric name (for axis labels).
    phenotype_col : str
        Phenotype name (for axis labels).

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
    sig_mask = df["p_value"] < 0.05

    fig = go.Figure()

    # Non-significant residues (grey fill)
    fig.add_trace(go.Scatter(
        x=df["residue_index"],
        y=df["correlation"],
        mode="lines",
        line=dict(color="lightgrey", width=1),
        fill="tozeroy",
        fillcolor="rgba(200,200,200,0.2)",
        name="p >= 0.05",
        hovertemplate=(
            "Residue %{x}<br>"
            "r = %{y:.3f}<br>"
            "<extra></extra>"
        ),
    ))

    # Significant residues (colored markers on top)
    if sig_mask.any():
        sig_df = df[sig_mask]
        fig.add_trace(go.Scatter(
            x=sig_df["residue_index"],
            y=sig_df["correlation"],
            mode="markers",
            marker=dict(
                size=6,
                color=sig_df["correlation"],
                colorscale="RdBu_r",
                cmin=-1, cmax=1,
                colorbar=dict(title="r", x=1.02),
                line=dict(width=0.5, color="black"),
            ),
            name="p < 0.05",
            customdata=np.column_stack([sig_df["p_value"], sig_df["n_sequences"]]),
            hovertemplate=(
                "Residue %{x}<br>"
                "r = %{y:.3f}<br>"
                "p = %{customdata[0]:.2e}<br>"
                "n = %{customdata[1]}<br>"
                "<extra></extra>"
            ),
        ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)

    fig.update_layout(
        title=f"Per-Residue Correlation: {metric} vs {phenotype_col}",
        xaxis_title="Residue Index",
        yaxis_title="Pearson r",
        yaxis=dict(range=[-1.05, 1.05]),
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
        height=500,
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

            # Annotation with r and p
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            fig.add_annotation(
                text=f"r = {r:.3f} ({stars})<br>p = {p:.2e}",
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

    df["stars"] = df["p_value"].apply(_stars)
    df["abs_r"] = df["correlation"].abs()
    df["color"] = df["correlation"].apply(lambda r: "#EF553B" if r < 0 else "#636EFA")
    df["label"] = df.apply(
        lambda row: f"r = {row['correlation']:.3f} {row['stars']}", axis=1
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df["aggregation"],
        x=df["correlation"],
        orientation="h",
        marker_color=df["color"],
        text=df["label"],
        textposition="outside",
        customdata=np.column_stack([df["p_value"], df["n"]]),
        hovertemplate=(
            "%{y}<br>"
            "r = %{x:.3f}<br>"
            "p = %{customdata[0]:.2e}<br>"
            "n = %{customdata[1]}<br>"
            "<extra></extra>"
        ),
    ))

    # Zero line
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)

    x_max = max(df["abs_r"].max() * 1.4, 0.2)
    fig.update_layout(
        title=f"Overall Correlation Summary: {metric} vs {phenotype_col}",
        xaxis_title="Pearson r",
        yaxis_title="Aggregation",
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

    # Tab 1: Per-residue correlation
    fig1 = plot_per_residue_correlation(per_residue_df, metric=metric, phenotype_col=phenotype_col)
    figures["Per-Residue Correlation"] = fig1

    # Tab 2: Scatter plots
    fig2 = plot_per_sequence_scatter(per_sequence_df, metric=metric, phenotype_col=phenotype_col,
                                     aggregations=aggregations)
    figures["Metric vs Phenotype"] = fig2

    # Tab 3: Summary
    fig3 = plot_overall_summary(overall_df, metric=metric, phenotype_col=phenotype_col)
    figures["Summary"] = fig3

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
    )


if __name__ == "__main__":
    main()
