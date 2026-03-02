#!/usr/bin/env python3
"""
Run ES (Effective Strain) analysis using PDAnalysis.

Processes sequences against a reference protein. Reads paths.txt files
(written by collect_cif_paths.py) to locate CIF files. Uses sys.executable
to guarantee the correct Python interpreter — no subprocess python mismatch.

Usage:
  # All sequences against a reference directory
  python run_es.py \
      --ref_dir /path/to/ref/boltz \
      --sequences_dir /path/to/sequences \
      --output_dir /path/to/es \
      --pdanalysis_dir /path/to/PDAnalysis \
      --method strain --min_plddt 70

  # All sequences against a single reference CIF
  python run_es.py \
      --ref_path /path/to/ref.cif \
      --sequences_dir /path/to/sequences \
      --output_dir /path/to/es \
      --pdanalysis_dir /path/to/PDAnalysis

  # Single sequence
  python run_es.py \
      --ref_dir /path/to/ref/boltz \
      --seq_dir /path/to/seq/boltz \
      --output /path/to/output.csv \
      --pdanalysis_dir /path/to/PDAnalysis
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def read_paths_txt(boltz_dir: Path) -> list[str]:
    """Read CIF paths from a paths.txt file in a boltz directory."""
    paths_file = boltz_dir / "paths.txt"
    if not paths_file.exists():
        return []
    return [line.strip() for line in paths_file.read_text().splitlines() if line.strip()]


def run_pdanalysis(
    pdanalysis_main: Path,
    ref_cifs: list[str],
    seq_cifs: list[str],
    output: Path,
    method: list[str],
    min_plddt: float,
    lddt_cutoffs: list[float],
    log_file: Path | None = None,
) -> bool:
    """Run PDAnalysis main.py using sys.executable (same python, no mismatch)."""
    cmd = [
        sys.executable, str(pdanalysis_main),
        "--protA", *ref_cifs,
        "--protB", *seq_cifs,
        "--method", *method,
        "--min_plddt", str(min_plddt),
        "--lddt_cutoffs", *[str(c) for c in lddt_cutoffs],
        "-o", str(output),
    ]

    output.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {"cwd": str(pdanalysis_main.parent)}
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            kwargs["stdout"] = f
            kwargs["stderr"] = subprocess.STDOUT
            result = subprocess.run(cmd, **kwargs)
    else:
        result = subprocess.run(cmd, **kwargs)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run ES analysis using PDAnalysis"
    )

    # Reference protein
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument("--ref_dir", help="Reference boltz dir (reads paths.txt)")
    ref_group.add_argument("--ref_path", help="Single reference CIF file")

    # Target sequences (mutually exclusive modes)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--sequences_dir", help="Process all sequences in directory")
    target.add_argument("--seq_dir", help="Process a single sequence boltz dir")

    # Output
    parser.add_argument("--output_dir", help="Output directory (with --sequences_dir)")
    parser.add_argument("--output", help="Output CSV path (with --seq_dir)")

    # PDAnalysis
    parser.add_argument(
        "--pdanalysis_dir", required=True,
        help="Path to PDAnalysis repo (containing main.py)",
    )

    # PDAnalysis options
    parser.add_argument("--method", nargs="+", default=["strain"])
    parser.add_argument("--min_plddt", type=float, default=70)
    parser.add_argument(
        "--lddt_cutoffs", nargs="+", type=float,
        default=[0.125, 0.25, 0.5, 1],
    )

    args = parser.parse_args()

    # Validate PDAnalysis
    pdanalysis_main = Path(args.pdanalysis_dir) / "main.py"
    if not pdanalysis_main.exists():
        print(f"ERROR: PDAnalysis main.py not found: {pdanalysis_main}", file=sys.stderr)
        sys.exit(1)

    print(f"Python: {sys.executable}")
    print(f"PDAnalysis: {pdanalysis_main}")

    # Resolve reference CIFs
    if args.ref_path:
        ref_path = Path(args.ref_path)
        if not ref_path.exists():
            print(f"ERROR: Reference CIF not found: {ref_path}", file=sys.stderr)
            sys.exit(1)
        ref_cifs = [str(ref_path.resolve())]
    else:
        ref_cifs = read_paths_txt(Path(args.ref_dir))
        if not ref_cifs:
            print(
                f"ERROR: No CIF paths in {args.ref_dir}/paths.txt "
                f"(run collect_cif_paths.py first)",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Reference: {len(ref_cifs)} CIF file(s)")

    # --- Single sequence mode ---
    if args.seq_dir:
        if not args.output:
            parser.error("--output required with --seq_dir")

        seq_cifs = read_paths_txt(Path(args.seq_dir))
        if not seq_cifs:
            print(
                f"ERROR: No CIF paths in {args.seq_dir}/paths.txt",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Sequence: {len(seq_cifs)} CIF file(s)")
        ok = run_pdanalysis(
            pdanalysis_main, ref_cifs, seq_cifs, Path(args.output),
            args.method, args.min_plddt, args.lddt_cutoffs,
        )
        sys.exit(0 if ok else 1)

    # --- Batch mode ---
    if not args.output_dir:
        parser.error("--output_dir required with --sequences_dir")

    sequences_dir = Path(args.sequences_dir)
    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Collect sequences that have paths.txt
    sequences = []
    for seq_dir in sorted(sequences_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        boltz_dir = seq_dir / "boltz"
        paths_file = boltz_dir / "paths.txt"
        if paths_file.exists():
            cifs = read_paths_txt(boltz_dir)
            if cifs:
                sequences.append((seq_dir.name, cifs))

    if not sequences:
        print(
            f"ERROR: No sequences with paths.txt found in {sequences_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    total = len(sequences)
    success = 0
    failed = 0
    skipped = 0

    for i, (seq_name, seq_cifs) in enumerate(sequences, 1):
        output_csv = output_dir / f"{seq_name}.csv"

        # Resume: skip if output already exists
        if output_csv.exists():
            print(f"[{i}/{total}] {seq_name} SKIP (already exists)")
            skipped += 1
            success += 1
            continue

        t0 = time.time()
        ok = run_pdanalysis(
            pdanalysis_main, ref_cifs, seq_cifs, output_csv,
            args.method, args.min_plddt, args.lddt_cutoffs,
            log_file=log_dir / f"{seq_name}.log",
        )
        elapsed = time.time() - t0

        if ok and output_csv.exists():
            print(f"[{i}/{total}] {seq_name} OK ({elapsed:.1f}s)")
            success += 1
        else:
            print(f"[{i}/{total}] {seq_name} FAILED ({elapsed:.1f}s) — see logs/{seq_name}.log")
            failed += 1

    print(f"\nSummary: {success}/{total} succeeded ({skipped} skipped), {failed} failed")

    if success == 0:
        print("ERROR: All sequences failed", file=sys.stderr)
        sys.exit(1)
    if failed > 0:
        print(f"WARNING: {failed} sequences failed — check {log_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
