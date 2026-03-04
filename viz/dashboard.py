#!/usr/bin/env python3
"""Build a multi-tab HTML dashboard from ProtForge pipeline outputs.

Produces a tabbed HTML with:
    1. Dashboard   -- all sequences ranked by mean strain, mutation annotations
    2. Confidence  -- pLDDT model confidence heatmap/profiles per sequence
    3. 3D Overlay  -- reference vs mutants (dropdown to switch target)

Usage:
    # From a ProtForge output directory (auto-discovers data)
    python -m viz.dashboard --output_dir /path/to/outputs --html dashboard.html

    # With explicit paths
    python -m viz.dashboard \
        --sequences_dir /path/to/sequences \
        --es_dir /path/to/es \
        --ref_cif /path/to/ref.cif \
        --html dashboard.html

    # Only specific tabs
    python -m viz.dashboard --output_dir /path/to/outputs --tabs dashboard confidence
"""

import argparse
import sys
from pathlib import Path

from .visualization import (
    plot_dashboard,
    plot_3d_overlay_multi,
    write_dashboard_html,
    _load_df_dict,
)
from ._cif_parser import (
    load_confidence_dfs,
    _find_best_cif,
    parse_cif,
)


def _find_cif_for_overlay(boltz_dir):
    """Return the first CIF path from a boltz dir (for 3D views)."""
    cif = _find_best_cif(Path(boltz_dir))
    return str(cif) if cif else None


def build_dashboard(
    sequences_dir,
    es_dir=None,
    ref_cif=None,
    output_html="dashboard.html",
    tabs=None,
    metric="strain",
    title="ProtForge Dashboard",
):
    """Build and write the multi-tab dashboard.

    Parameters
    ----------
    sequences_dir : str or Path
        ProtForge sequences directory.
    es_dir : str or Path, optional
        ES output directory (contains {name}.csv files).
    ref_cif : str or Path, optional
        Reference CIF file path (for 3D overlay).
    output_html : str or Path
        Output HTML file path.
    tabs : list[str], optional
        Which tabs to include. Default: all available.
        Options: "dashboard", "confidence", "3d_overlay"
    metric : str
        ES metric to visualise (default "strain").
    title : str
        HTML page title.
    """
    sequences_dir = Path(sequences_dir)
    if tabs is None:
        tabs = ["dashboard", "confidence", "3d_overlay"]

    figures = {}

    # --- Tab 1: ES Dashboard ---
    if "dashboard" in tabs and es_dir:
        es_dir = Path(es_dir)
        if es_dir.is_dir():
            csvs = list(es_dir.glob("*.csv"))
            if csvs:
                print(f"Loading {len(csvs)} ES CSVs from {es_dir} ...")
                fig_dash = plot_dashboard(
                    str(es_dir),
                    metric=metric,
                    sort_by="mean",
                    highlight_mutations=True,
                    log_y=True,
                    title=f"Effective Strain Dashboard ({metric})",
                )
                figures["Dashboard"] = fig_dash
                print("Dashboard tab ready.")
            else:
                print(f"No CSV files found in {es_dir}, skipping Dashboard tab.")
        else:
            print(f"ES directory not found: {es_dir}, skipping Dashboard tab.")

    # --- Tab 2: Confidence (pLDDT) ---
    if "confidence" in tabs and sequences_dir.is_dir():
        print(f"Loading pLDDT from CIF files in {sequences_dir} ...")
        conf_dfs = load_confidence_dfs(sequences_dir)
        if conf_dfs:
            # Enrich with mutation info from ES data (protA=ref, protB=target)
            has_mutations = False
            if es_dir:
                es_path = Path(es_dir)
                if es_path.is_dir():
                    try:
                        es_dfs = _load_df_dict(str(es_path))
                        for name in conf_dfs:
                            if name not in es_dfs:
                                continue
                            es_df = es_dfs[name]
                            if "protA_resname" not in es_df.columns:
                                continue
                            conf_df = conf_dfs[name]
                            # Current protA_resname is the target sequence; rename it
                            conf_df = conf_df.rename(
                                columns={"protA_resname": "protB_resname"}
                            )
                            # Add reference sequence from ES data
                            ref_map = dict(zip(
                                es_df["residue_index"].astype(int),
                                es_df["protA_resname"],
                            ))
                            conf_df["protA_resname"] = (
                                conf_df["residue_index"].map(ref_map)
                            )
                            conf_dfs[name] = conf_df
                            has_mutations = True
                    except FileNotFoundError:
                        pass

            print(f"  Found pLDDT data for {len(conf_dfs)} sequences.")
            fig_conf = plot_dashboard(
                conf_dfs,
                metric="plddt",
                sort_by="mean",
                highlight_mutations=has_mutations,
                log_y=False,
                cmap="RdYlGn",
                title="Model Confidence (pLDDT)",
            )
            figures["Confidence"] = fig_conf
            print("Confidence tab ready.")
        else:
            print("No CIF files with pLDDT found, skipping Confidence tab.")

    # --- Tab 3: 3D Multi-Overlay ---
    if "3d_overlay" in tabs and ref_cif and sequences_dir.is_dir():
        ref_cif = Path(ref_cif)
        if ref_cif.exists():
            # Discover target CIFs that also have ES data
            targets_cif = {}
            es_dir_path = Path(es_dir) if es_dir else None
            df_dict = None

            for seq_dir in sorted(sequences_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                name = seq_dir.name
                cif = _find_cif_for_overlay(seq_dir / "boltz")
                if cif:
                    # Include target if it has a CIF (ES CSV is optional)
                    targets_cif[name] = cif

            if targets_cif:
                # Load ES data for strain colouring (optional)
                if es_dir_path and es_dir_path.is_dir():
                    try:
                        df_dict = _load_df_dict(str(es_dir_path))
                    except FileNotFoundError:
                        df_dict = None

                # Load pLDDT data for confidence colouring
                conf_dict = load_confidence_dfs(sequences_dir)
                if not conf_dict:
                    conf_dict = None

                print(f"Building 3D overlay ({len(targets_cif)} targets) ...")
                fig_3d = plot_3d_overlay_multi(
                    str(ref_cif),
                    targets_cif,
                    df_dict=df_dict,
                    confidence_dict=conf_dict,
                    metric=metric,
                    label_A="Reference",
                    title="3D Overlay",
                )
                figures["3D Overlay"] = fig_3d
                print("3D Overlay tab ready.")
            else:
                print("No target CIF files found, skipping 3D Overlay tab.")
        else:
            print(f"Reference CIF not found: {ref_cif}, skipping 3D Overlay tab.")

    # --- Write HTML ---
    if figures:
        out = write_dashboard_html(figures, output_html, title=title)
        print(f"\nTabbed HTML written to: {out}")
        print("Open in a browser to explore.")
    else:
        print("No data found. Check paths above.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Build a multi-tab HTML dashboard from ProtForge outputs."
    )

    # Data sources
    parser.add_argument(
        "--output_dir",
        help="ProtForge output parent directory (auto-discovers sequences/ and es/)",
    )
    parser.add_argument(
        "--sequences_dir",
        help="Sequences directory (overrides --output_dir auto-discovery)",
    )
    parser.add_argument(
        "--es_dir",
        help="ES output directory (overrides --output_dir auto-discovery)",
    )
    parser.add_argument(
        "--ref_cif",
        help="Reference CIF file for 3D overlay",
    )
    parser.add_argument(
        "--config",
        help="Path to config.yaml (reads es.ref_path for reference CIF)",
    )

    # Output
    parser.add_argument(
        "--html", default="dashboard.html",
        help="Output HTML file path (default: dashboard.html)",
    )

    # Options
    parser.add_argument(
        "--tabs", nargs="+",
        choices=["dashboard", "confidence", "3d_overlay"],
        help="Which tabs to include (default: all available)",
    )
    parser.add_argument(
        "--metric", default="strain",
        help="ES metric to visualise (default: strain)",
    )
    parser.add_argument(
        "--title", default="ProtForge Dashboard",
        help="HTML page title",
    )

    # Dash server mode
    parser.add_argument(
        "--serve", action="store_true",
        help="Launch interactive Dash server (requires dash)",
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="Dash server port (default: 8050)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable Dash debug mode",
    )

    args = parser.parse_args()

    # Resolve paths
    sequences_dir = args.sequences_dir
    es_dir = args.es_dir
    ref_cif = args.ref_cif

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not sequences_dir:
            sequences_dir = str(output_dir / "sequences")
        if not es_dir:
            es_dir = str(output_dir / "es")

    # Try to read ref_cif from config.yaml
    if not ref_cif and args.config:
        try:
            import yaml
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            ref_cif = cfg.get("es", {}).get("ref_path", "")
            if not ref_cif:
                ref_seq = cfg.get("es", {}).get("ref_seq", "")
                if ref_seq and sequences_dir:
                    ref_cif = _find_cif_for_overlay(
                        Path(sequences_dir) / ref_seq / "boltz"
                    )
        except Exception:
            pass

    if not sequences_dir:
        parser.error("Provide --output_dir or --sequences_dir")

    # Dash server mode
    if args.serve:
        from .app import create_app
        app = create_app(
            sequences_dir, es_dir=es_dir, ref_cif=ref_cif,
            metric=args.metric, title=args.title, tabs=args.tabs,
        )
        print(f"Starting Dash server on http://0.0.0.0:{args.port}")
        print(f"For remote access: ssh -L {args.port}:localhost:{args.port} <cluster>")
        app.run(host="0.0.0.0", port=args.port, debug=args.debug)
        return

    # Default HTML output: save alongside sequences dir
    html_path = args.html
    if html_path == "dashboard.html" and not Path(html_path).is_absolute():
        if args.output_dir:
            html_path = str(Path(args.output_dir) / "dashboard.html")
        else:
            html_path = str(Path(sequences_dir).parent / "dashboard.html")

    build_dashboard(
        sequences_dir=sequences_dir,
        es_dir=es_dir,
        ref_cif=ref_cif,
        output_html=html_path,
        tabs=args.tabs,
        metric=args.metric,
        title=args.title,
    )


if __name__ == "__main__":
    main()
