#!/usr/bin/env python3
"""
Split YAML/FASTA file paths into chunk files for parallel ESM/ESMFold processing.

Default mode: writes id_0.txt, id_1.txt, ... — one chunk per --num_chunks.

ESMFold smart-binning mode (--bin_by_length): partitions sequences into a
"short" pool (L < --length_threshold) and a "long" pool (L >= threshold) and
chunks each pool independently into id_short_0.txt, ..., id_long_0.txt, ...
This lets the Snakemake rule request different SLURM resources per pool.

Optional --max_seq_len drops oversized sequences before chunking. Dropped
entries are recorded in skipped_sequences.tsv.

Always writes manifest.txt (chunk file paths) and chunk_stats.tsv (joinable
to benchmarks/ by chunk_id; gains a `pool` column).
"""

import argparse
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


def write_skipped_tsv(output_dir: Path, skipped: list[tuple[Path, int, str]]) -> Path:
    """Always written (header even if no skipped entries)."""
    path = output_dir / "skipped_sequences.tsv"
    with open(path, "w") as f:
        f.write("path\tlength\treason\n")
        for p, L, reason in skipped:
            f.write(f"{p}\t{L}\t{reason}\n")
    return path


def write_chunk_stats(
    output_dir: Path,
    chunks: list[tuple[str, list[tuple[Path, int]], str]],
) -> Path:
    """Write chunk_stats.tsv next to manifest.txt — join key for benchmarks/.

    `chunks` is a list of (chunk_id_str, [(path, length), ...], pool_label).
    `pool_label` is "short" / "long" when binning is on, else "".
    """
    stats_path = output_dir / "chunk_stats.tsv"
    with open(stats_path, "w") as sf:
        sf.write(
            "chunk_id\tpool\tnum_seqs\tmean_len\tmin_len\tp95_len\tmax_len\ttotal_residues\n"
        )
        for chunk_id, items, pool in chunks:
            lengths = [L for _, L in items if L > 0]
            if not lengths:
                sf.write(f"{chunk_id}\t{pool}\t0\t0\t0\t0\t0\t0\n")
                continue
            mean_len = sum(lengths) / len(lengths)
            sf.write(
                f"{chunk_id}\t{pool}\t{len(lengths)}\t{mean_len:.1f}\t"
                f"{min(lengths)}\t{_percentile(lengths, 0.95)}\t"
                f"{max(lengths)}\t{sum(lengths)}\n"
            )
    return stats_path


def _split_evenly(items: list, num_chunks: int) -> list[list]:
    """Split into <=num_chunks groups by ceil-division, dropping empty trailers."""
    if not items or num_chunks <= 0:
        return []
    num = min(num_chunks, len(items))
    chunk_size = (len(items) + num - 1) // num
    out = []
    for i in range(num):
        s = i * chunk_size
        e = min(s + chunk_size, len(items))
        if s < e:
            out.append(items[s:e])
    return out


def _is_managed_chunk_file(stem: str) -> bool:
    """Predicate for stale-cleanup: only delete files we own (id_<digits>,
    id_short_<digits>, id_long_<digits>). Anything else with id_* prefix is
    left alone."""
    suffix = stem.removeprefix("id_")
    if suffix.isdigit():
        return True
    for label in ("short_", "long_"):
        if suffix.startswith(label) and suffix[len(label):].isdigit():
            return True
    return False


def _write_chunk(
    output_path: Path, chunk_id: str, items: list[tuple[Path, int]]
) -> Path:
    chunk_path = output_path / f"id_{chunk_id}.txt"
    with open(chunk_path, "w") as f:
        for fp, _ in items:
            f.write(f"{fp.resolve()}\n")
    print(f"Wrote {len(items)} paths -> {chunk_path}")
    return chunk_path


def create_chunks(
    items: list[tuple[Path, int]],
    output_dir: str,
    num_chunks: int,
    *,
    bin_by_length: bool = False,
    length_threshold: int = 1200,
    num_chunks_short: int | None = None,
    num_chunks_long: int | None = None,
):
    """Split (path, length) pairs into chunk files.

    Returns (chunk_files, chunks_meta) where chunks_meta is the list passed
    to write_chunk_stats.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Reruns may shrink the chunk count or toggle binning. Remove previous
    # chunk files we own so shell wrappers that glob id_*.txt don't resubmit
    # stale work.
    for stale in output_path.glob("id_*.txt"):
        if _is_managed_chunk_file(stale.stem):
            stale.unlink()

    chunk_files: list[Path] = []
    chunks_meta: list[tuple[str, list[tuple[Path, int]], str]] = []

    if bin_by_length:
        ns = num_chunks_short if num_chunks_short is not None else num_chunks
        nl = num_chunks_long if num_chunks_long is not None else num_chunks
        short_items = [it for it in items if it[1] < length_threshold]
        long_items = [it for it in items if it[1] >= length_threshold]
        for label, pool_items, n in (
            ("short", short_items, ns),
            ("long", long_items, nl),
        ):
            for i, group in enumerate(_split_evenly(pool_items, n)):
                chunk_id = f"{label}_{i}"
                chunk_files.append(_write_chunk(output_path, chunk_id, group))
                chunks_meta.append((chunk_id, group, label))
    else:
        for i, group in enumerate(_split_evenly(items, num_chunks)):
            chunk_id = str(i)
            chunk_files.append(_write_chunk(output_path, chunk_id, group))
            chunks_meta.append((chunk_id, group, ""))

    return chunk_files, chunks_meta


def _write_binned_chunk(output_path: Path, chunk_id: int, items) -> Path:
    chunk_path = output_path / f"id_{chunk_id}.txt"
    with open(chunk_path, "w") as f:
        for fp, _ in items:
            f.write(f"{fp.resolve()}\n")
    return chunk_path


def _cleanup_stale(output_path: Path) -> None:
    for stale in output_path.glob("id_*.txt"):
        if _is_managed_chunk_file(stale.stem):
            stale.unlink()


def main():
    parser = argparse.ArgumentParser(description="Split file paths into chunks for a stage")
    parser.add_argument("--yaml_dir", required=True, help="Directory containing input files (recursive)")
    parser.add_argument("--output_dir", required=True, help="Output directory for chunk files")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks (legacy/unbinned mode)")
    parser.add_argument("--pattern", default="*.yaml",
                        help="Glob pattern to match (default: *.yaml; use *.fasta for fasta mode)")
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="Drop sequences longer than this (residues). Default: no cutoff.")
    parser.add_argument("--bin_by_length", action="store_true",
                        help="Legacy 2-pool short/long binning (superseded by --enable_binning).")
    parser.add_argument("--length_threshold", type=int, default=1200,
                        help="With --bin_by_length: sequences with L>=threshold go to long pool.")
    parser.add_argument("--num_chunks_short", type=int, default=None,
                        help="Chunk count for short pool (--bin_by_length only). Defaults to --num_chunks.")
    parser.add_argument("--num_chunks_long", type=int, default=None,
                        help="Chunk count for long pool (--bin_by_length only). Defaults to --num_chunks.")
    add_binning_argparse(parser)
    args = parser.parse_args()

    files = find_files(args.yaml_dir, args.pattern)
    if not files:
        print(f"ERROR: No {args.pattern} files found in {args.yaml_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} {args.pattern} files in {args.yaml_dir}")

    pairs = [(f, _length_for(f)) for f in files]

    skipped: list[tuple[Path, int, str]] = []
    surviving: list[tuple[Path, int]] = []
    if args.max_seq_len is not None:
        for f, L in pairs:
            if L > args.max_seq_len:
                skipped.append((f.resolve(), L, f"length>{args.max_seq_len}"))
            else:
                surviving.append((f, L))
    else:
        surviving = pairs

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.enable_binning:
        # Bin-aware path: per-bin chunk_size + per-chunk SLURM resources via chunks.tsv.
        _cleanup_stale(output_path)
        recipes = recipes_from_args(args)
        thresholds = parse_int_csv(args.bin_thresholds) if args.bin_mode == "thresholds" else None
        chunks, eff_thresholds = compute_bins(
            surviving,
            mode=args.bin_mode,
            num_bins=args.num_bins,
            thresholds=thresholds,
            recipes=recipes,
            chunks_per_bin=args.chunks_per_bin,
        )
        chunk_files: list[Path] = []
        chunks_meta: list[tuple[str, list[tuple[Path, int]], str]] = []
        for c in chunks:
            chunk_path = _write_binned_chunk(output_path, c.chunk_id, c.items)
            chunk_files.append(chunk_path)
            chunks_meta.append((str(c.chunk_id), c.items, f"bin{c.bin_idx}"))
            print(f"Wrote {c.num_seqs} paths -> {chunk_path} (bin {c.bin_idx}, "
                  f"mem={c.mem_mb}MB time={c.runtime_min}min)")

        manifest_path = output_path / "manifest.txt"
        with open(manifest_path, "w") as mf:
            for cf in chunk_files:
                mf.write(f"{cf.resolve()}\n")

        skipped_path = write_skipped_tsv(output_path, skipped)
        stats_path = write_chunk_stats(output_path, chunks_meta)
        chunks_tsv = write_chunks_tsv(output_path, chunks)

        print(f"Created {len(chunk_files)} bin-aware chunks in {output_path}")
        print(f"Effective thresholds: {eff_thresholds}")
        print(f"Manifest:    {manifest_path}")
        print(f"Skipped:     {skipped_path} ({len(skipped)} sequences)")
        print(f"Chunk stats: {stats_path}")
        print(f"chunks.tsv:  {chunks_tsv}")
        return

    # Legacy paths (unbinned or 2-pool short/long).
    chunk_files, chunks_meta = create_chunks(
        surviving,
        args.output_dir,
        args.num_chunks,
        bin_by_length=args.bin_by_length,
        length_threshold=args.length_threshold,
        num_chunks_short=args.num_chunks_short,
        num_chunks_long=args.num_chunks_long,
    )

    manifest_path = output_path / "manifest.txt"
    with open(manifest_path, "w") as mf:
        for cf in chunk_files:
            mf.write(f"{cf.resolve()}\n")

    skipped_path = write_skipped_tsv(output_path, skipped)
    stats_path = write_chunk_stats(output_path, chunks_meta)

    print(f"Created {len(chunk_files)} chunk files in {output_path}")
    print(f"Manifest:    {manifest_path}")
    print(f"Skipped:     {skipped_path} ({len(skipped)} sequences)")
    print(f"Chunk stats: {stats_path}")


if __name__ == "__main__":
    main()
