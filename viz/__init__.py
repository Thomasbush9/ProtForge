"""ProtForge visualization module.

Provides interactive dashboards and 3D protein structure plots for
ProtForge pipeline outputs (ES strain, Boltz pLDDT confidence).

Quick start:
    from viz import plot_dashboard, plot_confidence, write_dashboard_html

    # ES strain dashboard
    fig = plot_dashboard("/path/to/es/", metric="strain")
    fig.show()

    # pLDDT confidence 3D view
    fig = plot_confidence("/path/to/structure.cif")
    fig.show()

    # Full multi-tab dashboard
    python -m viz.dashboard --output_dir /path/to/outputs --html dashboard.html

Install dependencies:
    pip install "protforge[viz]"
"""

from .visualization import (
    plot_backbone,
    plot_strain,
    plot_confidence,
    plot_comparison,
    plot_dashboard,
    plot_3d_overlay,
    plot_3d_overlay_multi,
    get_available_metrics,
    write_dashboard_html,
)
from ._cif_parser import (
    ProteinStructure,
    parse_cif,
    load_plddt_df,
    load_confidence_dfs,
)
from .dashboard import build_dashboard
from .correlation import (
    plot_per_residue_correlation,
    plot_per_sequence_scatter,
    plot_overall_summary,
    build_correlation_report,
)

try:
    from .app import create_app
except ImportError:
    pass

__all__ = [
    # 3D plots
    "plot_backbone",
    "plot_strain",
    "plot_confidence",
    "plot_comparison",
    # Dashboards
    "plot_dashboard",
    "plot_3d_overlay",
    "plot_3d_overlay_multi",
    "get_available_metrics",
    "write_dashboard_html",
    "build_dashboard",
    # Data loading
    "ProteinStructure",
    "parse_cif",
    "load_plddt_df",
    "load_confidence_dfs",
    # Correlation plots
    "plot_per_residue_correlation",
    "plot_per_sequence_scatter",
    "plot_overall_summary",
    "build_correlation_report",
    # Dash app (optional, requires dash)
    "create_app",
]
