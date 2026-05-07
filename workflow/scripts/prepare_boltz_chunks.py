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

import yaml

from binning import (
    add_binning_argparse,
    compute_bins,
    parse_int_csv,
    recipes_from_args,
    write_chunks_tsv,
)


def find_yaml_files(search_dir: str) -> list[Path]:
    """Recursively find all .yaml files, sorted."""
    return sorted(Path(search_dir).rglob("*.yaml"))


def _yaml_seq_length(path: Path) -> int:
    """Return total residues across all proteins in a Boltz YAML, or 0 on failure."""
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return 0
    if not isinstance(data, dict):
        return 0
    sequences = data.get("sequences") or []
    total = 0
    for entry in sequences:
        if not isinstance(entry, dict):
            continue
        protein = entry.get("protein") or {}
        seq = protein.get("sequence")
        if isinstance(seq, str):
            total += len(seq)
    return total


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def write_skipped_tsv(output_dir: Path, skipped: list[tuple[Path, int, str]]) -> Path:
    """Always written (header even if no skipped entries)."""
    path = output_dir / "skipped_sequences.tsv"
    with open(path, "w") as f:
        f.write("path\tlength\treason\n")
        for p, L, reason in skipped:
            f.write(f"{p}\t{L}\t{reason}\n")
    return path


def write_chunk_stats(output_dir: Path, chunks: list[tuple[int, list[Path]]]) -> Path:
    """Write chunk_stats.tsv next to manifest.txt — join key for benchmarks/."""
    stats_path = output_dir / "chunk_stats.tsv"
    with open(stats_path, "w") as sf:
        sf.write("chunk_id\tnum_seqs\tmean_len\tmin_len\tp95_len\tmax_len\ttotal_residues\n")
        for chunk_id, paths in chunks:
            lengths = [_yaml_seq_length(p) for p in paths]
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
    chunks: list[tuple[int, list[Path]]] = []
    for i in range(num_chunks):
        start = i * max_files_per_job
        end = min(start + max_files_per_job, total)
        if start >= end:
            continue

        chunk_dir = output_path / f"chunk_{i}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_files = yaml_files[start:end]
        for fp in chunk_files:
            src = fp.resolve()
            dst = chunk_dir / fp.name
            # Remove existing symlink if present (for reruns)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src, dst)

        chunk_dirs.append(chunk_dir)
        chunks.append((i, chunk_files))
        print(f"Created chunk_{i} with {end - start} YAML symlinks")

    # Write manifest
    manifest_path = output_path / "manifest.txt"
    with open(manifest_path, "w") as mf:
        for cd in chunk_dirs:
            mf.write(f"{cd.resolve()}\n")

    stats_path = write_chunk_stats(output_path, chunks)

    print(f"Created {len(chunk_dirs)} chunks in {output_path}")
    print(f"Manifest:    {manifest_path}")
    print(f"Chunk stats: {stats_path}")
    return chunk_dirs


def _create_binned_chunks(
    surviving_pairs: list[tuple[Path, int]],
    output_path: Path,
    chunks,
) -> list[Path]:
    """Materialize bin-aware chunks: one chunk_N/ dir per ChunkMeta, symlinks
    inside. Cleans stale chunk_*/ dirs first."""
    for stale in output_path.glob("chunk_*"):
        if stale.is_dir() and stale.name.removeprefix("chunk_").isdigit():
            shutil.rmtree(stale)

    chunk_dirs: list[Path] = []
    for c in chunks:
        chunk_dir = output_path / f"chunk_{c.chunk_id}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for fp, _ in c.items:
            src = fp.resolve()
            dst = chunk_dir / fp.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src, dst)
        chunk_dirs.append(chunk_dir)
        print(f"Created chunk_{c.chunk_id} with {c.num_seqs} symlinks "
              f"(bin {c.bin_idx}, mem={c.mem_mb}MB time={c.runtime_min}min)")
    return chunk_dirs


def main():
    parser = argparse.ArgumentParser(description="Prepare Boltz chunk directories")
    parser.add_argument("--yaml_dir", required=True, help="Directory containing YAML files (recursive search)")
    parser.add_argument("--output_dir", required=True, help="Output directory for boltz_chunks/")
    parser.add_argument("--max_files_per_job", type=int, required=True, help="Max YAML files per chunk")
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="Drop sequences longer than this (residues). Default: no cutoff.")
    add_binning_argparse(parser)
    args = parser.parse_args()

    yaml_files = find_yaml_files(args.yaml_dir)
    if not yaml_files:
        print(f"ERROR: No .yaml files found in {args.yaml_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(yaml_files)} YAML files in {args.yaml_dir}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    skipped: list[tuple[Path, int, str]] = []
    surviving_pairs: list[tuple[Path, int]] = []
    if args.max_seq_len is not None:
        for f in yaml_files:
            L = _yaml_seq_length(f)
            if L > args.max_seq_len:
                skipped.append((f.resolve(), L, f"length>{args.max_seq_len}"))
            else:
                surviving_pairs.append((f, L))
        print(f"Filter --max_seq_len={args.max_seq_len}: kept {len(surviving_pairs)}, "
              f"skipped {len(skipped)}")
    else:
        surviving_pairs = [(f, _yaml_seq_length(f)) for f in yaml_files]

    write_skipped_tsv(output_path, skipped)

    if args.enable_binning:
        recipes = recipes_from_args(args)
        thresholds = parse_int_csv(args.bin_thresholds) if args.bin_mode == "thresholds" else None
        chunks, eff_thresholds = compute_bins(
            surviving_pairs,
            mode=args.bin_mode,
            num_bins=args.num_bins,
            thresholds=thresholds,
            recipes=recipes,
            chunks_per_bin=args.chunks_per_bin,
        )
        chunk_dirs = _create_binned_chunks(surviving_pairs, output_path, chunks)

        manifest_path = output_path / "manifest.txt"
        with open(manifest_path, "w") as mf:
            for cd in chunk_dirs:
                mf.write(f"{cd.resolve()}\n")

        # chunk_stats.tsv expected by calibrate analyzer (numeric IDs)
        legacy_chunks = [(c.chunk_id, [p for p, _ in c.items]) for c in chunks]
        stats_path = write_chunk_stats(output_path, legacy_chunks)
        chunks_tsv = write_chunks_tsv(output_path, chunks)

        print(f"Created {len(chunk_dirs)} bin-aware chunks in {output_path}")
        print(f"Effective thresholds: {eff_thresholds}")
        print(f"Manifest:    {manifest_path}")
        print(f"Chunk stats: {stats_path}")
        print(f"chunks.tsv:  {chunks_tsv}")
        return

    create_chunks(
        [p for p, _ in surviving_pairs], args.output_dir, args.max_files_per_job
    )


if __name__ == "__main__":
    main()
