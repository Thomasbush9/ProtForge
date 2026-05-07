#!/usr/bin/env python3
"""
Bin-aware chunking shared library.

Used by chunk_yamls_for_esm.py / prepare_boltz_chunks.py / chunk_fastas.py to
partition sequences by length, pack them into per-bin chunks, and emit a
`chunks.tsv` sidecar that the Snakemake rule's `resources:` callables read to
request per-chunk SLURM mem / runtime.

Bin policy
----------
Two modes:

  - quantile (default): bin boundaries are quantile cuts of the input length
    distribution. Default cuts are [q0, q25, q50, q75, q90, q95, q100] (5 bins,
    upper-tail-weighted to match where O(L²) memory bites — same shape as
    scripts/calibrate/subsample.py).

  - thresholds: explicit length cuts. `thresholds=[400, 800, 1200, 1800]`
    produces 5 bins (0-400, 400-800, 800-1200, 1200-1800, 1800+).

Per-bin recipe
--------------
Each bin needs a (chunk_size, mem_mb, runtime_min) triple. Caller passes them
as parallel lists; len(chunk_sizes) must equal the number of bins. The webapp
estimator populates these from scaling_models.yaml; CLI users write them by
hand in config.yaml under `<stage>.binning.bins`.

Output sidecar
--------------
chunks.tsv columns:
    chunk_id  bin  num_seqs  min_len  max_len  p95_len  mem_mb  runtime_min

Numeric chunk_ids (0, 1, 2, ...) sequential across all bins. The rule's
mem_mb / runtime callables look up chunks.tsv by chunk_id and return the
per-chunk values. Falls back to stage-level defaults if chunks.tsv missing
(i.e., binning disabled).
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_QUANTILE_CUTS: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)


@dataclass
class BinRecipe:
    """Per-bin chunking + resource recipe. One BinRecipe per bin."""
    chunk_size: int
    mem_mb: int
    runtime_min: int


@dataclass
class ChunkMeta:
    """One row of chunks.tsv."""
    chunk_id: int
    bin_idx: int
    items: list[tuple[Path, int]]  # (path, length)
    mem_mb: int
    runtime_min: int

    @property
    def num_seqs(self) -> int:
        return len(self.items)

    @property
    def lengths(self) -> list[int]:
        return [L for _, L in self.items if L > 0]


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def quantile_thresholds(lengths: list[int], num_bins: int) -> list[int]:
    """Compute (num_bins - 1) length cuts.

    For num_bins == 6 uses upper-tail-weighted cuts (q25, q50, q75, q90, q95)
    matching DEFAULT_QUANTILE_CUTS. For other num_bins falls back to
    evenly-spaced quantiles (q1/N, q2/N, ..., q(N-1)/N).
    """
    if not lengths or num_bins < 2:
        return []
    if num_bins == 6:
        cuts = DEFAULT_QUANTILE_CUTS[1:-1]  # 5 cuts: q25, q50, q75, q90, q95
    else:
        cuts = tuple((i + 1) / num_bins for i in range(num_bins - 1))
    return [_percentile(lengths, q) for q in cuts]


def assign_bins(
    items: list[tuple[Path, int]],
    thresholds: list[int],
) -> list[list[tuple[Path, int]]]:
    """Partition items into len(thresholds)+1 buckets by length cuts.

    Bucket i contains items with thresholds[i-1] <= L < thresholds[i],
    where threshold[-1] == -inf and threshold[N] == +inf.
    """
    n_bins = len(thresholds) + 1
    buckets: list[list[tuple[Path, int]]] = [[] for _ in range(n_bins)]
    for path, L in items:
        idx = 0
        for t in thresholds:
            if L < t:
                break
            idx += 1
        buckets[idx].append((path, L))
    return buckets


def pack_chunks(
    items_per_bin: list[list[tuple[Path, int]]],
    recipes: list[BinRecipe],
) -> list[ChunkMeta]:
    """Greedy-pack each bin's items into chunks of size recipe.chunk_size.

    Returns a flat list of ChunkMeta with sequential chunk_ids (0, 1, 2, ...).
    Empty bins produce no chunks.
    """
    if len(items_per_bin) != len(recipes):
        raise ValueError(
            f"Bin count mismatch: {len(items_per_bin)} bins vs {len(recipes)} recipes."
        )
    out: list[ChunkMeta] = []
    cid = 0
    for bin_idx, (bin_items, recipe) in enumerate(zip(items_per_bin, recipes)):
        if not bin_items:
            continue
        size = max(1, int(recipe.chunk_size))
        # Sort each bin by length so packing is deterministic and chunk
        # lengths are roughly contiguous within a bin.
        sorted_items = sorted(bin_items, key=lambda x: x[1])
        for i in range(0, len(sorted_items), size):
            group = sorted_items[i : i + size]
            out.append(ChunkMeta(
                chunk_id=cid,
                bin_idx=bin_idx,
                items=list(group),
                mem_mb=int(recipe.mem_mb),
                runtime_min=int(recipe.runtime_min),
            ))
            cid += 1
    return out


def parse_recipe_lists(
    chunk_sizes: list[int],
    mem_mb: list[int],
    runtime_min: list[int],
) -> list[BinRecipe]:
    """Validate three parallel lists and return a list of BinRecipe."""
    if not (len(chunk_sizes) == len(mem_mb) == len(runtime_min)):
        raise ValueError(
            f"Recipe lists must be the same length; got "
            f"chunk_sizes={len(chunk_sizes)} mem_mb={len(mem_mb)} "
            f"runtime_min={len(runtime_min)}"
        )
    return [
        BinRecipe(chunk_size=cs, mem_mb=m, runtime_min=r)
        for cs, m, r in zip(chunk_sizes, mem_mb, runtime_min)
    ]


def compute_bins(
    items: list[tuple[Path, int]],
    *,
    mode: str,
    num_bins: int,
    thresholds: list[int] | None,
    recipes: list[BinRecipe],
) -> tuple[list[ChunkMeta], list[int]]:
    """Top-level: compute thresholds, assign items, pack chunks.

    Returns (chunks, effective_thresholds). `effective_thresholds` is the
    actual cuts used (computed for quantile mode; echoed for threshold mode).
    """
    lengths = [L for _, L in items if L > 0]
    if mode == "quantile":
        if num_bins < 2:
            raise ValueError("quantile mode requires num_bins >= 2")
        eff = quantile_thresholds(lengths, num_bins)
    elif mode == "thresholds":
        if not thresholds:
            raise ValueError("thresholds mode requires --bin_thresholds")
        eff = sorted(int(t) for t in thresholds)
    else:
        raise ValueError(f"Unknown bin mode: {mode!r}")

    expected_n_bins = len(eff) + 1
    if len(recipes) != expected_n_bins:
        raise ValueError(
            f"Need {expected_n_bins} bin recipes (one per bin), got {len(recipes)}. "
            f"Thresholds: {eff}"
        )

    buckets = assign_bins(items, eff)
    return pack_chunks(buckets, recipes), eff


def write_chunks_tsv(output_dir: Path, chunks: list[ChunkMeta]) -> Path:
    """Emit chunks.tsv. Always written (header even with no chunks)."""
    path = output_dir / "chunks.tsv"
    with open(path, "w") as f:
        f.write(
            "chunk_id\tbin\tnum_seqs\tmin_len\tmax_len\tp95_len\t"
            "mem_mb\truntime_min\n"
        )
        for c in chunks:
            ls = c.lengths
            if ls:
                row = (
                    f"{c.chunk_id}\t{c.bin_idx}\t{c.num_seqs}\t{min(ls)}\t"
                    f"{max(ls)}\t{_percentile(ls, 0.95)}\t"
                    f"{c.mem_mb}\t{c.runtime_min}\n"
                )
            else:
                row = (
                    f"{c.chunk_id}\t{c.bin_idx}\t0\t0\t0\t0\t"
                    f"{c.mem_mb}\t{c.runtime_min}\n"
                )
            f.write(row)
    return path


def parse_int_csv(value: str | None) -> list[int]:
    """Parse '60,30,15,5,1' -> [60, 30, 15, 5, 1]. Empty string -> []."""
    if not value:
        return []
    return [int(x) for x in value.split(",") if x.strip()]


def add_binning_argparse(parser) -> None:
    """Attach --bin_* flags to a chunker's argparse.ArgumentParser."""
    parser.add_argument(
        "--enable_binning", action="store_true",
        help="Bin sequences by length and pack each bin into chunks "
             "with per-bin SLURM resources. Writes chunks.tsv.",
    )
    parser.add_argument(
        "--bin_mode", choices=["quantile", "thresholds"], default="quantile",
        help="quantile: cuts derived from input distribution. "
             "thresholds: explicit cuts passed via --bin_thresholds.",
    )
    parser.add_argument(
        "--num_bins", type=int, default=5,
        help="quantile mode only: number of bins.",
    )
    parser.add_argument(
        "--bin_thresholds", type=str, default="",
        help="thresholds mode only: comma-separated length cuts (e.g. '400,800,1200,1800').",
    )
    parser.add_argument(
        "--bin_chunk_sizes", type=str, default="",
        help="Comma-separated chunk_size per bin (one entry per bin, smallest first).",
    )
    parser.add_argument(
        "--bin_mem_mb", type=str, default="",
        help="Comma-separated mem_mb per bin.",
    )
    parser.add_argument(
        "--bin_runtime_min", type=str, default="",
        help="Comma-separated runtime_min per bin.",
    )


def recipes_from_args(args) -> list[BinRecipe]:
    return parse_recipe_lists(
        parse_int_csv(args.bin_chunk_sizes),
        parse_int_csv(args.bin_mem_mb),
        parse_int_csv(args.bin_runtime_min),
    )


# ---- chunks.tsv reader (used by Snakemake rule resources:) ----------------

def read_chunks_tsv(path: str | Path) -> dict[str, dict[str, str]]:
    """Read chunks.tsv. Returns {chunk_id_str: row_dict} or {} if missing."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row["chunk_id"]: row for row in reader}


def chunk_resource(
    chunks_tsv_path: str | Path,
    chunk_id: str | int,
    key: str,
    default,
):
    """Lookup `key` for `chunk_id` in chunks.tsv. Falls back to `default` if
    chunks.tsv doesn't exist or the row/key is missing.

    Designed to be called from Snakemake `resources:` callables — when binning
    is disabled, chunks.tsv won't exist and this returns `default`."""
    rows = read_chunks_tsv(chunks_tsv_path)
    row = rows.get(str(chunk_id))
    if not row:
        return default
    val = row.get(key)
    if val in (None, ""):
        return default
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def main():
    """CLI: print effective thresholds for a length list (debug helper)."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", required=True,
                        help="Comma-separated lengths to test (e.g. '100,200,500,1500')")
    add_binning_argparse(parser)
    args = parser.parse_args()
    if not args.enable_binning:
        print("--enable_binning not set; nothing to do.", file=sys.stderr)
        sys.exit(0)
    lengths = parse_int_csv(args.lengths)
    items = [(Path(f"seq_{i}"), L) for i, L in enumerate(lengths)]
    recipes = recipes_from_args(args)
    thresholds = parse_int_csv(args.bin_thresholds) if args.bin_mode == "thresholds" else None
    chunks, eff = compute_bins(
        items, mode=args.bin_mode, num_bins=args.num_bins,
        thresholds=thresholds, recipes=recipes,
    )
    print(f"Effective thresholds: {eff}")
    for c in chunks:
        ls = c.lengths
        rng = f"{min(ls)}-{max(ls)}" if ls else "-"
        print(
            f"  chunk_id={c.chunk_id} bin={c.bin_idx} n={c.num_seqs} "
            f"L={rng} mem={c.mem_mb} time={c.runtime_min}min"
        )


if __name__ == "__main__":
    main()
