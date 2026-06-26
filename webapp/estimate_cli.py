"""
Command-line front door to the ProtForge resource estimator.

Wraps the same pure functions the Streamlit UI uses (webapp/estimator.py,
webapp/validate.py) so a job can be sized and configured without the web app:

    python -m webapp.estimate_cli --config config.ga.yaml          # print table
    python -m webapp.estimate_cli --config config.ga.yaml --json    # machine-readable
    python -m webapp.estimate_cli --config config.ga.yaml --apply    # write resources back
    python -m webapp.estimate_cli --config config.ga.yaml --apply --dry-run  # preview, no write

Flow: read config -> pick the input dir (input.fasta_dir / input.yaml_dir, or
--input-dir) -> validate + summarize the sequences -> estimate every enabled
stage against the calibrated scaling models -> print, and optionally write the
recommendations back into the config (slurm.resources.<stage>, partitions,
<stage>.max_files_per_job, binning).

Designed to be driven by the `run-pipeline` Claude Code skill, but usable
standalone. Read-only unless --apply is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

# The webapp modules use flat imports (Streamlit runs from inside webapp/).
# Make that work whether this is invoked as `python -m webapp.estimate_cli`
# or `python webapp/estimate_cli.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from estimator import (  # noqa: E402
    CALIBRATED_SCALING_PATH,
    ALL_STAGES,
    compute_input_stats,
    estimate_all_stages,
    apply_estimate_to_config,
    load_scaling_models,
)
from validate import scan_directory  # noqa: E402


def _parse_gpu_prefs(values: list[str] | None) -> dict:
    """Turn --gpu args into a {stage: gpu_type} dict.

    Accepts either a bare GPU type (applies to every stage) or `stage=type`
    pairs. Bare and per-stage forms can be mixed; per-stage wins.
    Example: --gpu auto --gpu boltz=h100  ->  every stage auto, boltz pinned h100.
    """
    prefs: dict = {}
    bare: str | None = None
    for v in values or []:
        if "=" in v:
            stage, _, gpu = v.partition("=")
            prefs[stage.strip()] = gpu.strip()
        else:
            bare = v.strip()
    if bare is not None:
        for stage in ALL_STAGES:
            prefs.setdefault(stage, bare)
    return prefs


def _resolve_input_dir(cfg: dict, override: str | None) -> tuple[Path, str]:
    """Pick which directory to scan. Returns (path, why).

    Precedence: --input-dir > input.yaml_dir > input.fasta_dir, matching the
    pipeline's own yaml_dir > fasta_dir precedence for the sequence-only stages.
    """
    if override:
        return Path(override), "from --input-dir"
    inp = cfg.get("input", {}) or {}
    yaml_dir = inp.get("yaml_dir")
    fasta_dir = inp.get("fasta_dir")
    if yaml_dir and Path(yaml_dir).is_dir():
        return Path(yaml_dir), "from config input.yaml_dir"
    if fasta_dir and Path(fasta_dir).is_dir():
        return Path(fasta_dir), "from config input.fasta_dir"
    # Fall back to whatever is set even if missing, so the error is informative.
    if fasta_dir:
        return Path(fasta_dir), "from config input.fasta_dir (does not exist)"
    if yaml_dir:
        return Path(yaml_dir), "from config input.yaml_dir (does not exist)"
    raise SystemExit(
        "No input directory: set input.fasta_dir or input.yaml_dir in the "
        "config, or pass --input-dir."
    )


def _fmt_gb(mem_mb: int) -> str:
    return f"{mem_mb / 1024:.0f}G"


def _print_table(stats, estimates: dict, scaling_calibrated: bool) -> None:
    print(
        f"\nInputs: {stats.count} {stats.file_type} sequences | "
        f"len min/mean/p95/max = "
        f"{stats.min_len}/{stats.mean_len:.0f}/{stats.p95_len}/{stats.max_len} aa"
    )
    cal = ("scaling_models.yaml + .calibrated.yaml overlay"
           if scaling_calibrated else "scaling_models.yaml (no .calibrated.yaml overlay)")
    print(f"Scaling models: {cal}\n")

    header = ["stage", "gpu", "partition", "mem", "time", "cpu", "gpu#", "chunk", "chunks", "node·h"]
    rows = []
    for stage in ALL_STAGES:
        est = estimates.get(stage)
        if est is None:
            continue
        rows.append([
            stage,
            est.gpu_type or "cpu",
            est.partition or "-",
            _fmt_gb(est.mem_mb),
            f"{est.runtime_min}m",
            str(est.cpus),
            str(est.gpus),
            str(est.chunk_size),
            str(est.num_chunks),
            f"{est.total_node_hours:g}",
        ])

    if not rows:
        print("No enabled stages to estimate (check pipeline.* toggles).")
        return

    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(header))))
    for r in rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))

    total_nh = sum(e.total_node_hours for e in estimates.values())
    print(f"\nTotal estimated cost: {total_nh:.1f} node-hours")

    # Notes (OOM risk, binning, proxy GPU) are the important warnings.
    any_notes = False
    for stage in ALL_STAGES:
        est = estimates.get(stage)
        if est and est.notes:
            if not any_notes:
                print("\nNotes:")
                any_notes = True
            for n in est.notes:
                print(f"  [{stage}] {n}")
        if est and est.bin_plan is not None:
            print(f"  [{stage}] binning: {est.bin_plan.num_bins} bins, "
                  f"{est.bin_plan.total_chunks} chunks total")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="estimate_cli",
        description="Estimate ProtForge SLURM resources from calibrated scaling models.",
    )
    p.add_argument("--config", required=True, help="Path to the pipeline config YAML.")
    p.add_argument("--input-dir", help="Override the input dir to scan (else from config).")
    p.add_argument(
        "--gpu", action="append", metavar="[stage=]TYPE",
        help="GPU preference: bare TYPE (e.g. h100) for all stages, or stage=TYPE "
             "(e.g. boltz=h100). 'auto' lets the estimator pick. Repeatable.",
    )
    p.add_argument("--apply", action="store_true",
                   help="Write the recommendations back into the config.")
    p.add_argument("--dry-run", action="store_true",
                   help="With --apply, show what would change but do not write.")
    p.add_argument("--no-backup", action="store_true",
                   help="With --apply, skip writing a timestamped config backup.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of the table.")
    args = p.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    input_dir, why = _resolve_input_dir(cfg, args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found ({why}): {input_dir}")

    scan = scan_directory(input_dir)
    if scan["file_type"] == "yaml":
        stats = compute_input_stats(yaml_results=scan["yaml_results"])
    else:
        # fasta or mixed — estimate from fasta results (validate skips invalids)
        stats = compute_input_stats(fasta_results=scan["fasta_results"])

    if stats.count == 0:
        raise SystemExit(
            f"No valid sequences found in {input_dir} ({why}). "
            f"Scanned {scan['total_files']} files, {scan['invalid_count']} invalid."
        )

    gpu_prefs = _parse_gpu_prefs(args.gpu)
    scaling = load_scaling_models()
    estimates = estimate_all_stages(stats, cfg, scaling, gpu_prefs)

    scaling_calibrated = CALIBRATED_SCALING_PATH.exists()

    if args.json:
        payload = {
            "input_dir": str(input_dir),
            "input_source": why,
            "stats": stats.as_dict(),
            "scaling_calibrated": scaling_calibrated,
            "gpu_preferences": gpu_prefs,
            "estimates": {s: e.as_dict() for s, e in estimates.items()},
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"Input dir: {input_dir} ({why})")
        _print_table(stats, estimates, scaling_calibrated)

    if args.apply:
        if not estimates:
            print("\nNothing to apply (no enabled stages).", file=sys.stderr)
            return 1
        if args.dry_run:
            print(f"\n[dry-run] Would write resources for "
                  f"{', '.join(estimates)} into {config_path} (no changes made).")
        else:
            backup = apply_estimate_to_config(
                config_path, estimates, backup=not args.no_backup)
            print(f"\nApplied estimates to {config_path}")
            if not args.no_backup:
                print(f"Backup written to {backup}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
