#!/usr/bin/env python3
"""
Collect CIF file paths from Boltz output directories.

Scans sequence directories and writes paths.txt files listing all CIF paths.
Handles both multi-run (boltz/run_N/) and single-run (boltz/) layouts.

Usage:
  # Process all sequences in a directory
  python collect_cif_paths.py --sequences_dir /path/to/sequences

  # Process a single boltz directory
  python collect_cif_paths.py --boltz_dir /path/to/seq/boltz --output /path/to/paths.txt
"""

import argparse
import sys
from pathlib import Path


def discover_cifs(boltz_dir: Path) -> list[Path]:
    """Find best CIF files from a boltz output directory.

    Multi-run: scan run_*/, pick highest *_model_*.cif per run.
    Single-run: pick highest *_model_*.cif directly.
    """
    cifs = []
    run_dirs = sorted(boltz_dir.glob("run_*"))

    if run_dirs:
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            model_cifs = sorted(run_dir.glob("*_model_*.cif"))
            if model_cifs:
                cifs.append(model_cifs[-1].resolve())
            else:
                any_cifs = sorted(run_dir.glob("*.cif"))
                if any_cifs:
                    cifs.append(any_cifs[-1].resolve())
    else:
        model_cifs = sorted(boltz_dir.glob("*_model_*.cif"))
        if model_cifs:
            cifs.append(model_cifs[-1].resolve())
        else:
            any_cifs = sorted(boltz_dir.glob("*.cif"))
            if any_cifs:
                cifs.append(any_cifs[-1].resolve())

    return cifs


def write_paths(cifs: list[Path], output: Path) -> None:
    """Write CIF paths to a text file, one per line."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(str(c) for c in cifs) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Collect CIF paths from Boltz outputs"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--sequences_dir",
        help="Process all sequences in directory (writes {seq}/boltz/paths.txt each)",
    )
    mode.add_argument(
        "--boltz_dir",
        help="Process a single boltz directory",
    )

    parser.add_argument(
        "--output",
        help="Output path for paths.txt (required with --boltz_dir)",
    )

    args = parser.parse_args()

    if args.boltz_dir:
        if not args.output:
            parser.error("--output is required with --boltz_dir")

        boltz_dir = Path(args.boltz_dir)
        if not boltz_dir.is_dir():
            print(f"ERROR: Not a directory: {boltz_dir}", file=sys.stderr)
            sys.exit(1)

        cifs = discover_cifs(boltz_dir)
        if not cifs:
            print(f"WARNING: No CIF files found in {boltz_dir}", file=sys.stderr)
            sys.exit(0)

        write_paths(cifs, Path(args.output))
        print(f"Wrote {len(cifs)} CIF path(s) to {args.output}")
    else:
        sequences_dir = Path(args.sequences_dir)
        if not sequences_dir.is_dir():
            print(f"ERROR: Not a directory: {sequences_dir}", file=sys.stderr)
            sys.exit(1)

        total_seqs = 0
        total_cifs = 0

        for seq_dir in sorted(sequences_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            boltz_dir = seq_dir / "boltz"
            if not boltz_dir.is_dir():
                continue

            cifs = discover_cifs(boltz_dir)
            if not cifs:
                continue

            paths_file = boltz_dir / "paths.txt"
            write_paths(cifs, paths_file)
            total_seqs += 1
            total_cifs += len(cifs)

        print(f"Summary: {total_seqs} sequences, {total_cifs} total CIF files")


if __name__ == "__main__":
    main()
