#!/usr/bin/env python3
"""
Split FASTA files from an input directory into chunks for parallel MSA processing.

For each chunk, creates:
  - chunk_{i}/combined.fasta  (all sequences concatenated)
  - chunk_{i}/file_list.txt   (one original FASTA path per line)

Also writes msa_chunks/manifest.txt listing all chunk directories.

Extracted from: slurm_scripts/split_and_run_msa.sh
"""

import argparse
import os
import sys
from pathlib import Path


def find_fasta_files(input_dir: str) -> list[Path]:
    """Find all .fasta and .fa files in the input directory (non-recursive, sorted)."""
    input_path = Path(input_dir)
    files = sorted(
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix in (".fasta", ".fa")
    )
    return files


def create_chunks(fasta_files: list[Path], output_dir: str, max_files_per_job: int):
    """
    Split FASTA files into chunk directories.

    Each chunk directory contains:
      - combined.fasta: all sequences from the chunk concatenated
      - file_list.txt: absolute paths to original FASTA files in this chunk
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total = len(fasta_files)
    num_chunks = (total + max_files_per_job - 1) // max_files_per_job

    chunk_dirs = []
    for i in range(num_chunks):
        start = i * max_files_per_job
        end = min(start + max_files_per_job, total)
        if start >= end:
            continue

        chunk_dir = output_path / f"chunk_{i}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # Write file_list.txt with absolute paths
        file_list_path = chunk_dir / "file_list.txt"
        with open(file_list_path, "w") as fl:
            for j in range(start, end):
                fl.write(f"{fasta_files[j].resolve()}\n")

        # Write combined.fasta by concatenating all FASTA files in this chunk
        combined_path = chunk_dir / "combined.fasta"
        with open(combined_path, "w") as cf:
            for j in range(start, end):
                with open(fasta_files[j]) as f:
                    content = f.read()
                    cf.write(content)
                    if not content.endswith("\n"):
                        cf.write("\n")

        chunk_dirs.append(chunk_dir)
        print(f"Created chunk_{i} with {end - start} files")

    # Write manifest listing all chunk directories
    manifest_path = output_path / "manifest.txt"
    with open(manifest_path, "w") as mf:
        for cd in chunk_dirs:
            mf.write(f"{cd.resolve()}\n")

    print(f"Created {len(chunk_dirs)} chunks in {output_path}")
    print(f"Manifest: {manifest_path}")
    return chunk_dirs


def main():
    parser = argparse.ArgumentParser(description="Split FASTAs into chunks for MSA")
    parser.add_argument("--input_dir", required=True, help="Directory containing .fasta/.fa files")
    parser.add_argument("--output_dir", required=True, help="Output directory for msa_chunks/")
    parser.add_argument("--max_files_per_job", type=int, required=True, help="Max FASTA files per chunk")
    args = parser.parse_args()

    fasta_files = find_fasta_files(args.input_dir)
    if not fasta_files:
        print(f"ERROR: No .fasta or .fa files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(fasta_files)} FASTA files in {args.input_dir}")
    create_chunks(fasta_files, args.output_dir, args.max_files_per_job)


if __name__ == "__main__":
    main()
