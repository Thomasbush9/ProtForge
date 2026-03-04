"""Dash web app for ProtForge interactive dashboard.

Serves the same panels as the static HTML dashboard but renders
figures on-demand, handles filtering server-side, and keeps data
in server memory instead of embedding it in the browser.

Requires: pip install "protforge[dash]"
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context, no_update
    from dash.exceptions import PreventUpdate
except ImportError:
    raise ImportError(
        "dash is required for the interactive server. "
        "Install with: pip install 'protforge[dash]' or pip install dash"
    )

from .visualization import (
    plot_dashboard,
    plot_3d_overlay,
    _load_df_dict,
)
from ._cif_parser import (
    load_confidence_dfs,
    _find_best_cif,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enrich_confidence_dfs(conf_dfs, es_dfs):
    """Add mutation info (protA/protB resnames) from ES data into confidence DFs.

    Modifies *conf_dfs* in-place and returns whether mutations were found.
    """
    has_mutations = False
    for name in conf_dfs:
        if name not in es_dfs:
            continue
        es_df = es_dfs[name]
        if "protA_resname" not in es_df.columns:
            continue
        conf_df = conf_dfs[name]
        conf_df = conf_df.rename(columns={"protA_resname": "protB_resname"})
        ref_map = dict(zip(
            es_df["residue_index"].astype(int),
            es_df["protA_resname"],
        ))
        conf_df["protA_resname"] = conf_df["residue_index"].map(ref_map)
        conf_dfs[name] = conf_df
        has_mutations = True
    return has_mutations


def _filter_df_dict(df_dict, search=None, top_n=None, metric="strain",
                    sort_by="mean"):
    """Filter and sort a df_dict by search term and top-N."""
    if not df_dict:
        return {}

    names = list(df_dict.keys())

    # Filter by search term
    if search:
        search_lower = search.lower()
        names = [n for n in names if search_lower in n.lower()]

    if not names:
        return {}

    # Sort by mean metric value (descending)
    if sort_by:
        def _sort_key(n):
            df = df_dict[n]
            if metric in df.columns:
                vals = df[metric].dropna().values.astype(float)
                if len(vals):
                    if sort_by == "mean":
                        return float(np.nanmean(vals))
                    elif sort_by == "max":
                        return float(np.nanmax(vals))
                    elif sort_by == "median":
                        return float(np.nanmedian(vals))
            return 0.0
        names = sorted(names, key=_sort_key, reverse=True)

    # Slice to top_n
    if top_n and top_n < len(names):
        names = names[:top_n]

    return {n: df_dict[n] for n in names}


def _get_panel_ids(es_dfs, conf_dfs):
    """Return list of panel graph IDs that exist based on available data."""
    ids = []
    if es_dfs:
        ids.extend(["dash-heatmap", "dash-profiles",
                     "dash-distributions", "dash-summary"])
    if conf_dfs:
        ids.extend(["conf-heatmap", "conf-profiles",
                     "conf-distributions", "conf-summary"])
    return ids


def _discover_targets(sequences_dir):
    """Discover target CIF paths from sequences directory."""
    targets = {}
    sequences_dir = Path(sequences_dir)
    for seq_dir in sorted(sequences_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        cif = _find_best_cif(seq_dir / "boltz")
        if cif:
            targets[seq_dir.name] = str(cif)
    return targets


# ---------------------------------------------------------------------------
# CSS theme (reuses variables from static HTML dashboard)
# ---------------------------------------------------------------------------

_DASH_CSS = """
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

/* Header */
.dash-header {
    background: var(--bg-primary); border-bottom: 1px solid var(--border-color);
    padding: 16px 24px 0; flex-shrink: 0;
}
.dash-header h1 {
    margin: 0 0 12px; font-size: 20px; font-weight: 600;
    color: var(--text-primary); letter-spacing: -0.3px;
}

/* Control bar */
.control-bar {
    background: var(--bg-primary); padding: 10px 24px;
    border-bottom: 1px solid var(--border-color);
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
}
.control-bar label {
    font-size: 12px; color: var(--text-secondary);
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
}
.result-count {
    font-size: 12px; color: var(--text-secondary); margin-left: 4px;
}

/* Grid layout for panels */
.panel-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 5fr 3fr 2fr;
    height: calc(100vh - 200px);
    min-height: 900px;
    gap: 4px;
    padding: 8px 16px;
}
.panel-grid .panel-heatmap { grid-column: 1 / -1; }
.panel-grid .panel-summary { grid-column: 1 / -1; }

/* 3D overlay tab */
.overlay-controls {
    display: flex; gap: 16px; align-items: center;
    padding: 12px 24px; background: var(--bg-primary);
    border-bottom: 1px solid var(--border-color);
}
.overlay-controls label {
    font-size: 12px; color: var(--text-secondary);
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
}
.overlay-controls .Select {
    min-width: 240px;
}

/* Plotly graph sizing */
.dash-graph { height: 100%; }
.dash-graph .js-plotly-plot, .dash-graph .plotly { height: 100% !important; }

/* Tabs */
.custom-tabs-container { padding: 0 !important; }
.custom-tab {
    padding: 8px 20px !important; border: 1px solid var(--border-color) !important;
    border-bottom: none !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
    background: var(--bg-secondary) !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: var(--text-secondary) !important;
    font-family: var(--font) !important;
}
.custom-tab:hover { background: var(--bg-hover) !important; color: var(--text-primary) !important; }
.custom-tab--selected {
    background: var(--bg-primary) !important;
    border-bottom: 2px solid var(--bg-primary) !important;
    color: var(--accent) !important; font-weight: 600 !important;
}

/* Loading spinner */
._dash-loading { background: transparent !important; }
"""

# Clientside JS for cross-panel highlighting
_HIGHLIGHT_JS = """
(function() {
    var _highlighted = null;
    var _origData = {};

    function getPlots(tabId) {
        var tab = document.querySelector('[data-tab="' + tabId + '"]');
        if (!tab) return [];
        return Array.from(tab.querySelectorAll('.js-plotly-plot'));
    }

    function storeOriginals(plots) {
        plots.forEach(function(plot) {
            if (_origData[plot.id] || !plot.data) return;
            _origData[plot.id] = plot.data.map(function(t) {
                return { opacity: t.opacity, lineWidth: t.line ? t.line.width : undefined };
            });
        });
    }

    function applyHighlight(plots, seqName) {
        plots.forEach(function(plot) {
            if (!plot.data) return;
            storeOriginals([plot]);
            var updates = { opacity: [], 'line.width': [] };
            plot.data.forEach(function(t, i) {
                var meta = t.meta || {};
                var isTarget = meta.seqName === seqName;
                var isSpecial = (meta.seqName || '').indexOf('__') === 0;
                if (isSpecial) {
                    updates.opacity.push(t.opacity);
                    updates['line.width'].push(t.line ? t.line.width : undefined);
                } else {
                    updates.opacity.push(isTarget ? 1.0 : 0.08);
                    updates['line.width'].push(isTarget ? 3 : (t.line ? t.line.width : undefined));
                }
            });
            var indices = Array.from({length: plot.data.length}, function(_, i) { return i; });
            Plotly.restyle(plot, {opacity: [updates.opacity], 'line.width': [updates['line.width']]}, indices);
        });
    }

    function resetHighlight(plots) {
        plots.forEach(function(plot) {
            if (!plot.data || !_origData[plot.id]) return;
            var orig = _origData[plot.id];
            var updates = { opacity: [], 'line.width': [] };
            plot.data.forEach(function(t, i) {
                var o = orig[i] || {};
                updates.opacity.push(o.opacity !== undefined ? o.opacity : t.opacity);
                updates['line.width'].push(o.lineWidth !== undefined ? o.lineWidth : (t.line ? t.line.width : undefined));
            });
            var indices = Array.from({length: plot.data.length}, function(_, i) { return i; });
            Plotly.restyle(plot, {opacity: [updates.opacity], 'line.width': [updates['line.width']]}, indices);
        });
        _origData = {};
    }

    window._protforgeHighlight = {
        handleClick: function(clickData, tabId) {
            if (!clickData || !clickData.points || !clickData.points.length) return;
            var pt = clickData.points[0];
            var meta = pt.data ? pt.data.meta : null;
            var seqName = meta ? meta.seqName : null;
            if (!seqName || seqName.indexOf('__') === 0) return;

            var plots = getPlots(tabId);
            if (_highlighted === seqName) {
                _highlighted = null;
                resetHighlight(plots);
            } else {
                _highlighted = seqName;
                applyHighlight(plots, seqName);
            }
        },
        reset: function() {
            _highlighted = null;
            _origData = {};
        }
    };
})();
"""


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(sequences_dir, es_dir=None, ref_cif=None,
               metric="strain", title="ProtForge Dashboard",
               tabs=None):
    """Create and return a configured Dash application.

    Parameters
    ----------
    sequences_dir : str or Path
        ProtForge sequences directory.
    es_dir : str or Path, optional
        ES output directory (contains {name}.csv files).
    ref_cif : str or Path, optional
        Reference CIF file path (for 3D overlay).
    metric : str
        ES metric to visualise (default "strain").
    title : str
        Page title.
    tabs : list[str], optional
        Which tabs to include. Default: all available.
    """
    sequences_dir = Path(sequences_dir)
    if tabs is None:
        tabs = ["dashboard", "confidence", "3d_overlay"]

    # ------------------------------------------------------------------
    # Load data once at startup
    # ------------------------------------------------------------------

    # ES data (needed for dashboard tab, confidence enrichment, and 3D strain)
    es_dfs = {}
    if es_dir and ("dashboard" in tabs or "confidence" in tabs or "3d_overlay" in tabs):
        es_path = Path(es_dir)
        if es_path.is_dir():
            try:
                es_dfs = _load_df_dict(str(es_path))
                print(f"Loaded {len(es_dfs)} ES CSVs from {es_path}")
            except FileNotFoundError:
                print(f"No CSV files found in {es_path}")

    # Confidence (pLDDT) data
    conf_dfs = {}
    has_mutations = False
    if "confidence" in tabs and sequences_dir.is_dir():
        conf_dfs = load_confidence_dfs(sequences_dir)
        if conf_dfs and es_dfs:
            has_mutations = _enrich_confidence_dfs(conf_dfs, es_dfs)
        print(f"Loaded pLDDT data for {len(conf_dfs)} sequences")

    # 3D overlay targets
    targets_cif = {}
    if "3d_overlay" in tabs and sequences_dir.is_dir():
        targets_cif = _discover_targets(sequences_dir)
        # Auto-discover ref CIF if not provided: use first target
        if not ref_cif and targets_cif:
            first_name = next(iter(targets_cif))
            ref_cif = targets_cif.pop(first_name)
            print(f"Auto-selected reference CIF from '{first_name}'")
        elif ref_cif:
            ref_cif = str(Path(ref_cif))
        if targets_cif:
            print(f"Discovered {len(targets_cif)} target CIFs for 3D overlay")

    # ------------------------------------------------------------------
    # Build Dash app
    # ------------------------------------------------------------------

    app = dash.Dash(
        __name__,
        title=title,
        suppress_callback_exceptions=True,
    )

    # Inject custom CSS and JS
    app.index_string = f"""<!DOCTYPE html>
<html>
<head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{_DASH_CSS}</style>
</head>
<body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
    <script>{_HIGHLIGHT_JS}</script>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    # Build tab children
    tab_children = []

    if "dashboard" in tabs and es_dfs:
        tab_children.append(dcc.Tab(
            label="Dashboard",
            value="dashboard",
            className="custom-tab",
            selected_className="custom-tab--selected",
            children=html.Div([
                html.Div([
                    dcc.Graph(id="dash-heatmap", className="dash-graph panel-heatmap",
                              config={"displayModeBar": True, "scrollZoom": True}),
                    dcc.Graph(id="dash-profiles", className="dash-graph",
                              config={"displayModeBar": True, "scrollZoom": True}),
                    dcc.Graph(id="dash-distributions", className="dash-graph",
                              config={"displayModeBar": True, "scrollZoom": True}),
                    dcc.Graph(id="dash-summary", className="dash-graph panel-summary",
                              config={"displayModeBar": True, "scrollZoom": True}),
                ], className="panel-grid", **{"data-tab": "dashboard"}),
            ]),
        ))

    if "confidence" in tabs and conf_dfs:
        tab_children.append(dcc.Tab(
            label="Confidence",
            value="confidence",
            className="custom-tab",
            selected_className="custom-tab--selected",
            children=html.Div([
                html.Div([
                    dcc.Graph(id="conf-heatmap", className="dash-graph panel-heatmap",
                              config={"displayModeBar": True, "scrollZoom": True}),
                    dcc.Graph(id="conf-profiles", className="dash-graph",
                              config={"displayModeBar": True, "scrollZoom": True}),
                    dcc.Graph(id="conf-distributions", className="dash-graph",
                              config={"displayModeBar": True, "scrollZoom": True}),
                    dcc.Graph(id="conf-summary", className="dash-graph panel-summary",
                              config={"displayModeBar": True, "scrollZoom": True}),
                ], className="panel-grid", **{"data-tab": "confidence"}),
            ]),
        ))

    if "3d_overlay" in tabs and targets_cif and ref_cif:
        target_options = [{"label": n, "value": n} for n in sorted(targets_cif.keys())]
        metric_options = [
            {"label": "Effective Strain", "value": "strain"},
            {"label": "pLDDT Confidence", "value": "plddt"},
        ]
        default_target = target_options[0]["value"] if target_options else None
        tab_children.append(dcc.Tab(
            label="3D Overlay",
            value="3d_overlay",
            className="custom-tab",
            selected_className="custom-tab--selected",
            children=html.Div([
                html.Div([
                    html.Label("Target:"),
                    dcc.Dropdown(
                        id="3d-target-dropdown",
                        options=target_options,
                        value=default_target,
                        clearable=False,
                        style={"minWidth": "240px"},
                    ),
                    html.Label("Colouring:"),
                    dcc.Dropdown(
                        id="3d-metric-dropdown",
                        options=metric_options,
                        value="strain",
                        clearable=False,
                        style={"minWidth": "180px"},
                    ),
                ], className="overlay-controls"),
                dcc.Graph(
                    id="3d-overlay-graph",
                    className="dash-graph",
                    style={"height": "calc(100vh - 250px)"},
                    config={"displayModeBar": True, "scrollZoom": True},
                ),
            ]),
        ))

    if not tab_children:
        app.layout = html.Div([
            html.H1(title),
            html.P("No data found. Check your paths."),
        ])
        return app

    # Top-N options
    n_seqs = max(len(es_dfs), len(conf_dfs), 1)
    topn_options = [{"label": "All", "value": 0}]
    for n in [10, 20, 30, 50, 100]:
        if n < n_seqs:
            topn_options.append({"label": f"Top {n}", "value": n})
    default_tab = tab_children[0].value

    app.layout = html.Div([
        # Header
        html.Div([html.H1(title)], className="dash-header"),

        # Control bar
        html.Div([
            html.Label("Search:"),
            dcc.Input(
                id="search-input", type="text",
                placeholder="Filter sequences...",
                debounce=True,
                style={"width": "220px", "padding": "6px 10px",
                       "border": "1px solid #e2e5ea", "borderRadius": "8px",
                       "fontSize": "13px", "background": "#f8f9fb"},
            ),
            html.Label("Show:"),
            dcc.Dropdown(
                id="topn-dropdown",
                options=topn_options,
                value=0,
                clearable=False,
                style={"width": "120px"},
            ),
            html.Span(id="result-count", className="result-count"),
        ], className="control-bar"),

        # Tabs
        dcc.Tabs(
            id="main-tabs",
            value=default_tab,
            children=tab_children,
            className="custom-tabs-container",
        ),

        # Hidden stores for clientside highlight callbacks
        dcc.Store(id="highlighted-seq", data=None),
        *[dcc.Store(id=f"_click-sink-{pid}", data=None)
          for pid in _get_panel_ids(es_dfs, conf_dfs)],
    ])

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    # Result count — single callback, dispatches based on active tab
    @app.callback(
        Output("result-count", "children"),
        [Input("main-tabs", "value"),
         Input("search-input", "value"),
         Input("topn-dropdown", "value")],
        prevent_initial_call=False,
    )
    def update_result_count(active_tab, search, top_n):
        top_n = top_n or None
        if active_tab == "dashboard" and es_dfs:
            filtered = _filter_df_dict(es_dfs, search=search, top_n=top_n,
                                       metric=metric, sort_by="mean")
            return f"{len(filtered)} / {len(es_dfs)} sequences"
        elif active_tab == "confidence" and conf_dfs:
            filtered = _filter_df_dict(conf_dfs, search=search, top_n=top_n,
                                       metric="plddt", sort_by="mean")
            return f"{len(filtered)} / {len(conf_dfs)} sequences"
        return ""

    # Dashboard tab panels
    if es_dfs:
        @app.callback(
            [Output("dash-heatmap", "figure"),
             Output("dash-profiles", "figure"),
             Output("dash-distributions", "figure"),
             Output("dash-summary", "figure")],
            [Input("main-tabs", "value"),
             Input("search-input", "value"),
             Input("topn-dropdown", "value")],
            prevent_initial_call=False,
        )
        def update_dashboard(active_tab, search, top_n):
            if active_tab != "dashboard":
                raise PreventUpdate

            top_n = top_n or None
            filtered = _filter_df_dict(es_dfs, search=search, top_n=top_n,
                                       metric=metric, sort_by="mean")

            if not filtered:
                empty = {"data": [], "layout": {"template": "plotly_white",
                         "annotations": [{"text": "No matching sequences",
                          "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
                          "showarrow": False, "font": {"size": 16}}]}}
                return empty, empty, empty, empty

            figs = {}
            for panel in ["heatmap", "profiles", "distributions", "summary"]:
                fig = plot_dashboard(
                    filtered,
                    metric=metric,
                    sort_by="mean",
                    highlight_mutations=True,
                    log_y=True,
                    panels=[panel],
                    title="",
                )
                fig.update_layout(
                    height=None, width=None, autosize=True,
                    margin=dict(l=60, r=20, t=30, b=40),
                )
                figs[panel] = fig

            return (figs["heatmap"], figs["profiles"],
                    figs["distributions"], figs["summary"])

    # Confidence tab panels
    if conf_dfs:
        @app.callback(
            [Output("conf-heatmap", "figure"),
             Output("conf-profiles", "figure"),
             Output("conf-distributions", "figure"),
             Output("conf-summary", "figure")],
            [Input("main-tabs", "value"),
             Input("search-input", "value"),
             Input("topn-dropdown", "value")],
            prevent_initial_call=False,
        )
        def update_confidence(active_tab, search, top_n):
            if active_tab != "confidence":
                raise PreventUpdate

            top_n = top_n or None
            filtered = _filter_df_dict(conf_dfs, search=search, top_n=top_n,
                                       metric="plddt", sort_by="mean")

            if not filtered:
                empty = {"data": [], "layout": {"template": "plotly_white",
                         "annotations": [{"text": "No matching sequences",
                          "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
                          "showarrow": False, "font": {"size": 16}}]}}
                return empty, empty, empty, empty

            figs = {}
            for panel in ["heatmap", "profiles", "distributions", "summary"]:
                fig = plot_dashboard(
                    filtered,
                    metric="plddt",
                    sort_by="mean",
                    highlight_mutations=has_mutations,
                    log_y=False,
                    cmap="RdYlGn",
                    panels=[panel],
                    title="",
                )
                fig.update_layout(
                    height=None, width=None, autosize=True,
                    margin=dict(l=60, r=20, t=30, b=40),
                )
                figs[panel] = fig

            return (figs["heatmap"], figs["profiles"],
                    figs["distributions"], figs["summary"])

    # 3D overlay
    if targets_cif and ref_cif:
        @app.callback(
            Output("3d-overlay-graph", "figure"),
            [Input("3d-target-dropdown", "value"),
             Input("3d-metric-dropdown", "value")],
            prevent_initial_call=False,
        )
        def update_3d_overlay(target_name, color_metric):
            if not target_name or target_name not in targets_cif:
                raise PreventUpdate

            target_cif = targets_cif[target_name]

            # Build strain/plddt values array
            strain_values = None
            colorbar_label = "Effective Strain"
            cmap = "Viridis"

            if color_metric == "strain" and target_name in es_dfs:
                es_df = es_dfs[target_name]
                if metric in es_df.columns:
                    from ._cif_parser import parse_cif as _parse
                    ref_prot = _parse(ref_cif)
                    n_res = len(ref_prot.sequence) if ref_prot.sequence is not None else 0
                    if n_res:
                        from .visualization import _df_metric_to_array
                        strain_values = _df_metric_to_array(es_df, metric, n_res)
                        colorbar_label = metric.replace("_", " ").title()
            elif color_metric == "plddt" and target_name in conf_dfs:
                conf_df = conf_dfs[target_name]
                if "plddt" in conf_df.columns:
                    from ._cif_parser import parse_cif as _parse
                    ref_prot = _parse(ref_cif)
                    n_res = len(ref_prot.sequence) if ref_prot.sequence is not None else 0
                    if n_res:
                        from .visualization import _df_metric_to_array
                        strain_values = _df_metric_to_array(conf_df, "plddt", n_res)
                        colorbar_label = "pLDDT"
                        cmap = "RdYlGn"

            fig = plot_3d_overlay(
                ref_cif, target_cif,
                strain_values=strain_values,
                label_A="Reference",
                label_B=target_name,
                title=f"3D Overlay: Reference vs {target_name}",
                cmap=cmap,
                colorbar_label=colorbar_label,
            )
            fig.update_layout(
                height=None, width=None, autosize=True,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            return fig

    # Clientside callbacks for cross-panel highlighting on click
    for panel_id in _get_panel_ids(es_dfs, conf_dfs):
        tab_prefix = panel_id.split("-")[0]
        tab_id = "dashboard" if tab_prefix == "dash" else "confidence"

        app.clientside_callback(
            f"""
            function(clickData) {{
                if (clickData) {{
                    window._protforgeHighlight.handleClick(clickData, '{tab_id}');
                }}
                return null;
            }}
            """,
            Output(f"_click-sink-{panel_id}", "data"),
            Input(panel_id, "clickData"),
            prevent_initial_call=True,
        )

    return app
