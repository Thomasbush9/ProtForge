"""
Command-line front door to the ProtForge results reader.

Wraps the same pure functions the Streamlit Results tab uses (webapp/results.py)
so a finished run can be inspected without the web app:

    python -m webapp.results_cli --config config.ga.yaml             # both summaries
    python -m webapp.results_cli --output-dir /path/to/outputs       # explicit dir
    python -m webapp.results_cli --config config.ga.yaml --json      # machine-readable
    python -m webapp.results_cli --config config.ga.yaml --benchmarks # only run analytics
    python -m webapp.results_cli --config config.ga.yaml --structures # only confidences

Flow: resolve the output dir (output.parent_dir from --config, or --output-dir)
-> read per-sequence structure confidences (pLDDT / pTM / ipTM / ranking) and
the per-stage Snakemake benchmark TSVs (runtime, peak RSS, node-hours) -> print
two ascii tables, or JSON.

Designed to be driven by the `results` Claude Code skill, but usable standalone.
Strictly read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

# The webapp modules use flat imports (Streamlit runs from inside webapp/).
# Make that work whether this is invoked as `python -m webapp.results_cli`
# or `python webapp/results_cli.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from results import (  # noqa: E402
    list_sequence_dirs,
    find_structures,
    model_summary,
    read_benchmarks,
    sequences_root,
    benchmarks_dir,
)

# Headline confidence fields, in display order. (label, summary key, format).
_CONF_FIELDS = [
    ("pLDDT", "plddt_mean", "{:.1f}"),
    ("pTM", "ptm", "{:.3f}"),
    ("ipTM", "iptm", "{:.3f}"),
    ("ranking", "ranking_score", "{:.3f}"),
]


def _resolve_output_dir(cfg: dict, override: str | None) -> tuple[Path, str]:
    """Pick the output tree to read. Returns (path, why)."""
    if override:
        return Path(override), "from --output-dir"
    out = (cfg.get("output", {}) or {}).get("parent_dir")
    if out:
        return Path(out), "from config output.parent_dir"
    raise SystemExit(
        "No output directory: set output.parent_dir in the config, or pass "
        "--output-dir."
    )


def _fmt(value, spec: str) -> str:
    if value is None:
        return "-"
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def collect_structures(output_dir: Path) -> list[dict]:
    """One row per (sequence, structure model) with headline confidences."""
    rows: list[dict] = []
    for seq in list_sequence_dirs(output_dir):
        for s in find_structures(sequences_root(output_dir) / seq):
            try:
                summ = model_summary(s)
            except Exception as exc:  # never let one bad file kill the table
                summ = {"_error": str(exc)}
            row = {
                "sequence": seq,
                "stage": s.stage,
                "model": s.path.name,
                "path": str(s.path),
            }
            for _, key, _ in _CONF_FIELDS:
                row[key] = summ.get(key)
            rows.append(row)
    return rows


def _print_structures(output_dir: Path, rows: list[dict]) -> None:
    n_seq = len({r["sequence"] for r in rows})
    print(f"\nStructures: {len(rows)} models across {n_seq} sequences")
    if not rows:
        root = sequences_root(output_dir)
        if not root.is_dir():
            print(f"  (no sequences/ dir under {output_dir} yet)")
        else:
            print("  (no predicted structures yet — only embeddings, or still "
                  "running)")
        return

    header = ["sequence", "stage", "model"] + [lbl for lbl, _, _ in _CONF_FIELDS]
    table = []
    for r in rows:
        table.append(
            [r["sequence"], r["stage"], r["model"]]
            + [_fmt(r[key], spec) for _, key, spec in _CONF_FIELDS]
        )
    _emit_table(header, table)


def _print_benchmarks(output_dir: Path, benches: dict) -> None:
    print(f"\nPer-stage benchmarks ({len(benches)} stages):")
    if not benches:
        bench = benchmarks_dir(output_dir)
        if not bench.is_dir():
            print(f"  (no benchmarks/ dir under {output_dir} yet)")
        else:
            print("  (no benchmark TSVs found)")
        return

    header = ["stage", "jobs", "node·h", "mean", "p95", "max", "peak mem"]
    table = []
    for b in benches.values():
        table.append([
            b.stage,
            str(b.n_jobs),
            f"{b.node_hours:.2f}",
            f"{b.mean_s / 60:.1f}m",
            f"{b.p95_s / 60:.1f}m",
            f"{b.max_s / 60:.1f}m",
            f"{b.max_rss_mb / 1024:.1f}G",
        ])
    _emit_table(header, table)

    total_nh = sum(b.node_hours for b in benches.values())
    total_jobs = sum(b.n_jobs for b in benches.values())
    print(f"\nTotal: {total_jobs} jobs, {total_nh:.1f} node-hours")


def _emit_table(header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("  ".join("-" * widths[i] for i in range(len(header))))
    for r in rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="results_cli",
        description="Summarize a finished ProtForge run: structure confidences "
                    "and per-stage runtime/memory benchmarks.",
    )
    p.add_argument("--config", help="Path to the pipeline config YAML "
                                    "(reads output.parent_dir).")
    p.add_argument("--output-dir", help="Output tree to read (overrides config).")
    p.add_argument("--structures", action="store_true",
                   help="Show only the structures/confidence table.")
    p.add_argument("--benchmarks", action="store_true",
                   help="Show only the per-stage benchmark table.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of tables.")
    args = p.parse_args(argv)

    cfg: dict = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            raise SystemExit(f"Config not found: {config_path}")
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    output_dir, why = _resolve_output_dir(cfg, args.output_dir)

    # Which sections to render. Neither flag => both.
    want_struct = args.structures or not args.benchmarks
    want_bench = args.benchmarks or not args.structures

    if not output_dir.is_dir():
        msg = f"Output directory not found ({why}): {output_dir}"
        if args.json:
            print(json.dumps({"output_dir": str(output_dir), "exists": False,
                              "message": msg}, indent=2))
        else:
            print(msg)
            print("Nothing to summarize yet — run the pipeline first.")
        return 0

    rows = collect_structures(output_dir) if want_struct else None
    benches = read_benchmarks(output_dir) if want_bench else None

    if args.json:
        payload: dict = {"output_dir": str(output_dir), "source": why, "exists": True}
        if rows is not None:
            payload["structures"] = rows
        if benches is not None:
            payload["benchmarks"] = {s: b.as_dict() for s, b in benches.items()}
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(f"Output dir: {output_dir} ({why})")
    if want_struct:
        _print_structures(output_dir, rows or [])
    if want_bench:
        _print_benchmarks(output_dir, benches or {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
