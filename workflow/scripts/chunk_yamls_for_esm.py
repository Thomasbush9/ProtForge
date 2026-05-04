#!/usr/bin/env python3
"""
Split YAML file paths into N chunk files for parallel ESM processing.

Writes id_0.txt, id_1.txt, ... each containing absolute paths to YAML files.
Also writes a manifest.txt listing all chunk files.

Extracted from: slurm_scripts/run_esm.sh
"""

import argparse
import sys
from pathlib import Path

import yaml


def find_files(search_dir: str, pattern: str = "*.yaml") -> list[Path]:
    """Recursively find files matching pattern, following symlinks, sorted."""
    return sorted(Path(search_dir).rglob(pattern))


def _yaml_seq_length(path: Path) -> int:
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


def _fasta_seq_length(path: Path) -> int:
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


def _length_for(path: Path) -> int:
    if path.suffix in (".yaml", ".yml"):
        return _yaml_seq_length(path)
    if path.suffix in (".fasta", ".fa"):
        return _fasta_seq_length(path)
    return 0


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def write_chunk_stats(output_dir: Path, chunks: list[tuple[int, list[Path]]]) -> Path:
    """Write chunk_stats.tsv next to manifest.txt — join key for benchmarks/."""
    stats_path = output_dir / "chunk_stats.tsv"
    with open(stats_path, "w") as sf:
        sf.write("chunk_id\tnum_seqs\tmean_len\tmin_len\tp95_len\tmax_len\ttotal_residues\n")
        for chunk_id, paths in chunks:
            lengths = [_length_for(p) for p in paths]
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


def create_chunks(yaml_files: list[Path], output_dir: str, num_chunks: int):
    """
    Split YAML paths into N chunk files (id_0.txt, id_1.txt, ...).
    Uses ceil-division so all files are covered.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Reruns may shrink the chunk count. Remove previous chunk files so shell
    # wrappers that glob id_*.txt do not resubmit stale work.
    for stale_chunk in output_path.glob("id_*.txt"):
        suffix = stale_chunk.stem.removeprefix("id_")
        if suffix.isdigit():
            stale_chunk.unlink()

    total = len(yaml_files)
    # Adjust num_chunks if more than total files
    num_chunks = min(num_chunks, total)
    chunk_size = (total + num_chunks - 1) // num_chunks

    chunk_files = []
    chunks: list[tuple[int, list[Path]]] = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        if start >= end:
            continue

        chunk_path = output_path / f"id_{i}.txt"
        with open(chunk_path, "w") as f:
            for j in range(start, end):
                f.write(f"{yaml_files[j].resolve()}\n")

        chunk_files.append(chunk_path)
        chunks.append((i, list(yaml_files[start:end])))
        print(f"Wrote {end - start} paths -> {chunk_path}")

    # Write manifest
    manifest_path = output_path / "manifest.txt"
    with open(manifest_path, "w") as mf:
        for cf in chunk_files:
            mf.write(f"{cf.resolve()}\n")

    stats_path = write_chunk_stats(output_path, chunks)

    print(f"Created {len(chunk_files)} chunk files in {output_path}")
    print(f"Manifest:    {manifest_path}")
    print(f"Chunk stats: {stats_path}")
    return chunk_files


def main():
    parser = argparse.ArgumentParser(description="Split file paths into chunks for a stage")
    parser.add_argument("--yaml_dir", required=True, help="Directory containing input files (recursive)")
    parser.add_argument("--output_dir", required=True, help="Output directory for chunk files")
    parser.add_argument("--num_chunks", type=int, required=True, help="Number of chunks to create")
    parser.add_argument("--pattern", default="*.yaml",
                        help="Glob pattern to match (default: *.yaml; use *.fasta for fasta mode)")
    args = parser.parse_args()

    files = find_files(args.yaml_dir, args.pattern)
    if not files:
        print(f"ERROR: No {args.pattern} files found in {args.yaml_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} {args.pattern} files in {args.yaml_dir}")
    create_chunks(files, args.output_dir, args.num_chunks)


if __name__ == "__main__":
    main()
