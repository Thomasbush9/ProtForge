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
import shutil
import os
import sys
from pathlib import Path

from binning import (
    add_binning_argparse,
    compute_bins,
    parse_int_csv,
    recipes_from_args,
    write_chunks_tsv,
)


def find_fasta_files(input_dir: str) -> list[Path]:
    """Find all .fasta and .fa files in the input directory (non-recursive, sorted)."""
    input_path = Path(input_dir)
    files = sorted(
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix in (".fasta", ".fa")
    )
    return files


def _fasta_residue_count(path: Path) -> int:
    """Sum residues across all sequences in a FASTA file."""
    total = 0
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(">"):
                    total += len(line.replace(" ", ""))
    except OSError:
        return 0
    return total


def _percentile(values: list[int], q: float) -> int:
    """Inclusive integer percentile — returns the value at index ceil(q*(n-1))."""
    if not values:
        return 0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def write_chunk_stats(
    output_dir: Path,
    chunks: list[tuple[int, list[Path]]],
) -> Path:
    """Write chunk_stats.tsv next to manifest.txt — join key for benchmarks/.

    Schema:
        chunk_id  num_seqs  mean_len  min_len  p95_len  max_len  total_residues
    """
    stats_path = output_dir / "chunk_stats.tsv"
    with open(stats_path, "w") as sf:
        sf.write("chunk_id\tnum_seqs\tmean_len\tmin_len\tp95_len\tmax_len\ttotal_residues\n")
        for chunk_id, paths in chunks:
            lengths = [_fasta_residue_count(p) for p in paths]
            lengths = [L for L in lengths if L > 0]
            if not lengths:
                sf.write(f"{chunk_id}\t0\t0\t0\t0\t0\t0\n")
                continue
            mean_len = sum(lengths) / len(lengths)
            sf.write(
                f"{chunk_id}\t{len(lengths)}\t{mean_len:.1f}\t{min(lengths)}\t"
                f"{_percentile(lengths, 0.95)}\t{max(lengths)}\t{sum(lengths)}\n"
            )
    return stats_path


def create_chunks(fasta_files: list[Path], output_dir: str, max_files_per_job: int):
    """
    Split FASTA files into chunk directories.

    Each chunk directory contains:
      - combined.fasta: all sequences from the chunk concatenated
      - file_list.txt: absolute paths to original FASTA files in this chunk
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Reruns may reuse the same output directory. Remove previous chunk
    # directories so stale FASTAs and downstream MSA outputs cannot survive.
    for stale_dir in output_path.glob("chunk_*"):
        if stale_dir.is_dir() and stale_dir.name.removeprefix("chunk_").isdigit():
            shutil.rmtree(stale_dir)

    total = len(fasta_files)
    num_chunks = (total + max_files_per_job - 1) // max_files_per_job

    chunk_dirs = []
    chunks: list[tuple[int, list[Path]]] = []
    for i in range(num_chunks):
        start = i * max_files_per_job
        end = min(start + max_files_per_job, total)
        if start >= end:
            continue

        chunk_dir = output_path / f"chunk_{i}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_files = fasta_files[start:end]

        # Write file_list.txt with absolute paths
        file_list_path = chunk_dir / "file_list.txt"
        with open(file_list_path, "w") as fl:
            for fp in chunk_files:
                fl.write(f"{fp.resolve()}\n")

        # Write combined.fasta by concatenating all FASTA files in this chunk
        combined_path = chunk_dir / "combined.fasta"
        with open(combined_path, "w") as cf:
            for fp in chunk_files:
                with open(fp) as f:
                    content = f.read()
                    cf.write(content)
                    if not content.endswith("\n"):
                        cf.write("\n")

        chunk_dirs.append(chunk_dir)
        chunks.append((i, chunk_files))
        print(f"Created chunk_{i} with {end - start} files")

    # Write manifest listing all chunk directories
    manifest_path = output_path / "manifest.txt"
    with open(manifest_path, "w") as mf:
        for cd in chunk_dirs:
            mf.write(f"{cd.resolve()}\n")

    stats_path = write_chunk_stats(output_path, chunks)

    print(f"Created {len(chunk_dirs)} chunks in {output_path}")
    print(f"Manifest:    {manifest_path}")
    print(f"Chunk stats: {stats_path}")
    return chunk_dirs


def _materialize_chunk(chunk_dir: Path, fasta_paths: list[Path]) -> None:
    """Write file_list.txt + combined.fasta into chunk_dir."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    file_list_path = chunk_dir / "file_list.txt"
    with open(file_list_path, "w") as fl:
        for fp in fasta_paths:
            fl.write(f"{fp.resolve()}\n")
    combined_path = chunk_dir / "combined.fasta"
    with open(combined_path, "w") as cf:
        for fp in fasta_paths:
            with open(fp) as f:
                content = f.read()
                cf.write(content)
                if not content.endswith("\n"):
                    cf.write("\n")


def _create_binned_chunks(output_path: Path, chunks) -> list[Path]:
    """One chunk_N/ dir per ChunkMeta with combined.fasta + file_list.txt.
    Cleans stale chunk_*/ dirs first."""
    for stale in output_path.glob("chunk_*"):
        if stale.is_dir() and stale.name.removeprefix("chunk_").isdigit():
            shutil.rmtree(stale)

    chunk_dirs: list[Path] = []
    for c in chunks:
        chunk_dir = output_path / f"chunk_{c.chunk_id}"
        _materialize_chunk(chunk_dir, [p for p, _ in c.items])
        chunk_dirs.append(chunk_dir)
        print(f"Created chunk_{c.chunk_id} with {c.num_seqs} files "
              f"(bin {c.bin_idx}, mem={c.mem_mb}MB time={c.runtime_min}min)")
    return chunk_dirs


def main():
    parser = argparse.ArgumentParser(description="Split FASTAs into chunks for MSA")
    parser.add_argument("--input_dir", required=True, help="Directory containing .fasta/.fa files")
    parser.add_argument("--output_dir", required=True, help="Output directory for msa_chunks/")
    parser.add_argument("--max_files_per_job", type=int, required=True, help="Max FASTA files per chunk")
    add_binning_argparse(parser)
    args = parser.parse_args()

    fasta_files = find_fasta_files(args.input_dir)
    if not fasta_files:
        print(f"ERROR: No .fasta or .fa files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(fasta_files)} FASTA files in {args.input_dir}")

    if args.enable_binning:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pairs: list[tuple[Path, int]] = [(f, _fasta_residue_count(f)) for f in fasta_files]
        recipes = recipes_from_args(args)
        thresholds = parse_int_csv(args.bin_thresholds) if args.bin_mode == "thresholds" else None
        chunks, eff_thresholds = compute_bins(
            pairs,
            mode=args.bin_mode,
            num_bins=args.num_bins,
            thresholds=thresholds,
            recipes=recipes,
        )
        chunk_dirs = _create_binned_chunks(output_path, chunks)

        manifest_path = output_path / "manifest.txt"
        with open(manifest_path, "w") as mf:
            for cd in chunk_dirs:
                mf.write(f"{cd.resolve()}\n")

        legacy_chunks = [(c.chunk_id, [p for p, _ in c.items]) for c in chunks]
        stats_path = write_chunk_stats(output_path, legacy_chunks)
        chunks_tsv = write_chunks_tsv(output_path, chunks)

        print(f"Created {len(chunk_dirs)} bin-aware chunks in {output_path}")
        print(f"Effective thresholds: {eff_thresholds}")
        print(f"Manifest:    {manifest_path}")
        print(f"Chunk stats: {stats_path}")
        print(f"chunks.tsv:  {chunks_tsv}")
        return

    create_chunks(fasta_files, args.output_dir, args.max_files_per_job)


if __name__ == "__main__":
    main()
