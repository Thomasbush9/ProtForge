#!/usr/bin/env python3
"""
Compute ES (Effective Strain) for a single sequence by averaging across
multiple Boltz prediction runs.

Discovers CIF files in run_N/ subdirectories and calls PDAnalysis main.py
with --protA (reference) and --protB (sequence) arguments. When multiple
CIF files are found, PDAnalysis uses AverageProtein internally.

Usage:
    python compute_es.py \
        --ref_dir /path/to/ref/boltz \
        --seq_dir /path/to/seq/boltz \
        --output /path/to/output.csv \
        --pdanalysis_dir /path/to/PDAnalysis \
        --method strain \
        --min_plddt 70
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def discover_cifs(directory: str) -> list[str]:
    """Find all model CIF files in run_N/ subdirectories.

    Looks for patterns like run_0/*_model_*.cif, run_1/*_model_*.cif, etc.
    Falls back to *.cif directly in the directory if no run_N/ subdirs exist.
    """
    d = Path(directory)
    cifs = []

    # Multi-run: run_0/, run_1/, ...
    run_dirs = sorted(d.glob("run_*"))
    if run_dirs:
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            model_cifs = sorted(run_dir.glob("*_model_*.cif"))
            if model_cifs:
                # Take the highest model number (last after sorting)
                cifs.append(str(model_cifs[-1].resolve()))
            else:
                # Fallback: any CIF in the run dir
                any_cifs = sorted(run_dir.glob("*.cif"))
                if any_cifs:
                    cifs.append(str(any_cifs[-1].resolve()))
    else:
        # Single-run: CIFs directly in the directory
        model_cifs = sorted(d.glob("*_model_*.cif"))
        if model_cifs:
            cifs.append(str(model_cifs[-1].resolve()))
        else:
            any_cifs = sorted(d.glob("*.cif"))
            if any_cifs:
                cifs.append(str(any_cifs[-1].resolve()))

    return cifs


def main():
    parser = argparse.ArgumentParser(
        description="Compute ES for a single sequence, averaging across Boltz runs."
    )

    # Reference protein (mutually exclusive)
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument(
        "--ref_dir",
        help="Reference boltz dir with run_N/ subdirs (uses AverageProtein)",
    )
    ref_group.add_argument(
        "--ref_path",
        help="Single reference CIF file (uses Protein)",
    )

    # Target sequence
    parser.add_argument(
        "--seq_dir", required=True,
        help="Sequence boltz dir with run_N/ subdirs",
    )

    # Output
    parser.add_argument(
        "--output", required=True,
        help="Output CSV path",
    )

    # PDAnalysis location
    parser.add_argument(
        "--pdanalysis_dir", required=True,
        help="Path to PDAnalysis repo (containing main.py)",
    )

    # PDAnalysis options
    parser.add_argument(
        "--method", nargs="+", default=["strain"],
        help="Deformation methods (default: strain)",
    )
    parser.add_argument(
        "--min_plddt", type=float, default=70,
        help="pLDDT cutoff (default: 70)",
    )
    parser.add_argument(
        "--lddt_cutoffs", nargs="+", type=float, default=[0.125, 0.25, 0.5, 1],
        help="LDDT distance thresholds",
    )

    args = parser.parse_args()

    # Resolve reference CIF paths
    if args.ref_path:
        ref_cifs = [str(Path(args.ref_path).resolve())]
        if not Path(args.ref_path).exists():
            print(f"ERROR: Reference CIF not found: {args.ref_path}", file=sys.stderr)
            sys.exit(1)
    else:
        ref_cifs = discover_cifs(args.ref_dir)
        if not ref_cifs:
            print(f"ERROR: No CIF files found in ref_dir: {args.ref_dir}", file=sys.stderr)
            sys.exit(1)

    # Resolve sequence CIF paths
    seq_cifs = discover_cifs(args.seq_dir)
    if not seq_cifs:
        print(f"ERROR: No CIF files found in seq_dir: {args.seq_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Reference: {len(ref_cifs)} CIF file(s)")
    for c in ref_cifs:
        print(f"  {c}")
    print(f"Sequence:  {len(seq_cifs)} CIF file(s)")
    for c in seq_cifs:
        print(f"  {c}")

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Build PDAnalysis main.py command
    pdanalysis_main = Path(args.pdanalysis_dir) / "main.py"
    if not pdanalysis_main.exists():
        print(f"ERROR: PDAnalysis main.py not found at: {pdanalysis_main}", file=sys.stderr)
        sys.exit(1)

    # Use 'python' from PATH (respects conda env) rather than sys.executable
    # which may point to a different interpreter
    python_bin = shutil.which("python") or sys.executable
    cmd = [
        python_bin, str(pdanalysis_main),
        "--protA", *ref_cifs,
        "--protB", *seq_cifs,
        "--method", *args.method,
        "--min_plddt", str(args.min_plddt),
        "--lddt_cutoffs", *[str(c) for c in args.lddt_cutoffs],
        "-o", args.output,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=args.pdanalysis_dir)

    if result.returncode != 0:
        print(f"ERROR: PDAnalysis exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    if Path(args.output).exists():
        print(f"Output written to: {args.output}")
    else:
        print(f"WARNING: Expected output file not found: {args.output}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
