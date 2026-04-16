#!/usr/bin/env python3
"""
Prepare Boltz chunk directories by symlinking YAML files into chunk dirs.

Scans for .yaml files in either:
  - sequences_dir (after MSA stage: sequences/{name}/{name}.yaml)
  - yaml_dir (when MSA is skipped: user-provided directory with .yaml files)

Groups them into chunk directories for parallel boltz predict.

Extracted from: slurm_scripts/split_and_run_boltz.sh
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def find_yaml_files(search_dir: str) -> list[Path]:
    """Recursively find all .yaml files, sorted."""
    return sorted(Path(search_dir).rglob("*.yaml"))


def create_chunks(yaml_files: list[Path], output_dir: str, max_files_per_job: int):
    """
    Create chunk directories with symlinked YAML files.

    Each chunk directory contains symlinks to the original YAML files,
    since boltz predict takes a directory of YAMLs as input.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Reruns may target an existing boltz_chunks directory. Remove only the
    # input chunk directories (chunk_N) so stale YAML symlinks do not leak into
    # the next submission, while preserving run outputs like chunk_N_run_M_output.
    for stale_dir in output_path.glob("chunk_*"):
        suffix = stale_dir.name.removeprefix("chunk_")
        if stale_dir.is_dir() and suffix.isdigit():
            shutil.rmtree(stale_dir)

    total = len(yaml_files)
    num_chunks = (total + max_files_per_job - 1) // max_files_per_job

    chunk_dirs = []
    for i in range(num_chunks):
        start = i * max_files_per_job
        end = min(start + max_files_per_job, total)
        if start >= end:
            continue

        chunk_dir = output_path / f"chunk_{i}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        for j in range(start, end):
            src = yaml_files[j].resolve()
            dst = chunk_dir / yaml_files[j].name
            # Remove existing symlink if present (for reruns)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src, dst)

        chunk_dirs.append(chunk_dir)
        print(f"Created chunk_{i} with {end - start} YAML symlinks")

    # Write manifest
    manifest_path = output_path / "manifest.txt"
    with open(manifest_path, "w") as mf:
        for cd in chunk_dirs:
            mf.write(f"{cd.resolve()}\n")

    print(f"Created {len(chunk_dirs)} chunks in {output_path}")
    print(f"Manifest: {manifest_path}")
    return chunk_dirs


def main():
    parser = argparse.ArgumentParser(description="Prepare Boltz chunk directories")
    parser.add_argument("--yaml_dir", required=True, help="Directory containing YAML files (recursive search)")
    parser.add_argument("--output_dir", required=True, help="Output directory for boltz_chunks/")
    parser.add_argument("--max_files_per_job", type=int, required=True, help="Max YAML files per chunk")
    args = parser.parse_args()

    yaml_files = find_yaml_files(args.yaml_dir)
    if not yaml_files:
        print(f"ERROR: No .yaml files found in {args.yaml_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(yaml_files)} YAML files in {args.yaml_dir}")
    create_chunks(yaml_files, args.output_dir, args.max_files_per_job)


if __name__ == "__main__":
    main()
