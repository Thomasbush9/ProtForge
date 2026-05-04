#!/usr/bin/env python3
"""
Stratified sample of FASTAs by sequence length for a calibration sweep.

Picks N files spanning the length distribution of a source directory, with
extra weight on the long tail (which dominates GPU memory for O(L^2) models).
The resulting calibration set is what `slurm_scripts/calibrate.sh` runs against
to produce real cluster numbers per (stage, GPU type).

Usage:
    python scripts/calibrate/subsample.py \
        --input_dir /path/to/7k_fastas \
        --output_dir tests/calibration_inputs/fastas \
        --n 20 \
        --seed 42

Writes:
    output_dir/<original filenames>     copies of the selected FASTAs
    output_dir/manifest.csv             filename, length, bin, source_path
"""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

# Quantile boundaries — extra resolution on the upper tail because O(L^2)
# memory blows up there and that is what the calibration must capture.
DEFAULT_BINS = (0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)


def parse_fasta_length(path: Path) -> int | None:
    """Return total residues across all sequences in path, or None on failure."""
    try:
        text = path.read_text()
    except OSError:
        return None
    if not text.strip():
        return None
    total = 0
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith(">"):
            total += len(line.replace(" ", ""))
    return total if total > 0 else None


def stratified_sample(
    paths_with_len: list[tuple[Path, int]],
    n: int,
    seed: int,
    bins: tuple[float, ...] = DEFAULT_BINS,
) -> list[tuple[Path, int, str]]:
    """Pick up to n items spanning quantile bins of length.

    Returns sorted list of (path, length, bin_label).
    Bins are *quantile* edges of the input length distribution, so the
    sampler self-adjusts to whatever range the user has.
    """
    if not paths_with_len or n <= 0:
        return []

    sorted_items = sorted(paths_with_len, key=lambda x: x[1])
    total = len(sorted_items)
    rng = random.Random(seed)

    bucket_labels: list[str] = []
    bucket_items: list[list[tuple[Path, int]]] = []

    for i in range(len(bins) - 1):
        lo_q, hi_q = bins[i], bins[i + 1]
        lo_idx = int(lo_q * total)
        hi_idx = max(lo_idx + 1, int(hi_q * total))
        slice_items = sorted_items[lo_idx:hi_idx]
        if not slice_items:
            continue
        label = f"q{int(lo_q * 100):02d}-q{int(hi_q * 100):02d}"
        bucket_labels.append(label)
        bucket_items.append(slice_items)

    if not bucket_items:
        return []

    # Distribute n across non-empty buckets; push leftover into the upper bins.
    base = n // len(bucket_items)
    leftover = n - base * len(bucket_items)
    quotas = [base] * len(bucket_items)
    for i in range(leftover):
        quotas[-(i % len(bucket_items)) - 1] += 1

    picked: list[tuple[Path, int, str]] = []
    for label, items, quota in zip(bucket_labels, bucket_items, quotas):
        take = min(quota, len(items))
        for path, L in rng.sample(items, take):
            picked.append((path, L, label))

    picked.sort(key=lambda x: x[1])
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="Directory of .fasta/.fa files to sample from")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Where to place the calibration set")
    parser.add_argument("--n", type=int, default=20,
                        help="Total samples to pick (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducible sampling (default: 42)")
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy",
                        help="copy is portable across mounts; symlink is faster (default: copy)")
    args = parser.parse_args()

    fastas = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix in (".fasta", ".fa")
    )
    if not fastas:
        print(f"No .fasta/.fa files in {args.input_dir}", file=sys.stderr)
        return 1

    print(f"Scanning {len(fastas)} files for sequence length...")
    paths_with_len: list[tuple[Path, int]] = []
    skipped = 0
    for p in fastas:
        L = parse_fasta_length(p)
        if L is None:
            skipped += 1
            continue
        paths_with_len.append((p, L))

    if skipped:
        print(f"  skipped {skipped} unreadable/empty files")

    if not paths_with_len:
        print("No usable FASTAs — nothing to sample.", file=sys.stderr)
        return 1

    picked = stratified_sample(paths_with_len, args.n, args.seed)
    if not picked:
        print("Sample came back empty — check --n and input distribution.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "length", "bin", "source_path"])
        for src, L, label in picked:
            dest = args.output_dir / src.name
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            if args.mode == "symlink":
                dest.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dest)
            w.writerow([src.name, L, label, str(src.resolve())])

    print(f"\nWrote {len(picked)} samples to {args.output_dir}")
    print(f"Manifest:        {manifest_path}")

    by_bin: dict[str, list[int]] = {}
    for _, L, label in picked:
        by_bin.setdefault(label, []).append(L)
    print("\nLength distribution of sampled set:")
    for label in sorted(by_bin):
        Ls = sorted(by_bin[label])
        print(f"  {label:<12}  n={len(Ls):>2}  range=[{Ls[0]:>5}-{Ls[-1]:>5}]")
    all_L = [L for _, L, _ in picked]
    print(f"\nOverall: min={min(all_L)} max={max(all_L)} "
          f"median={sorted(all_L)[len(all_L) // 2]} n={len(all_L)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
