#!/usr/bin/env python3
"""
Pick the "staircase" of FASTAs for the stair benchmark.

For each target length in a fixed arithmetic ladder (e.g. 200, 400, ..., 3000),
select the *real* sequence from --input_dir whose length is closest to the
target, with no sequence reused across rungs. Real sequences are used (not
synthetic) so MSA depth and residue composition are realistic — the VRAM we
later measure is then the true worst-case the GPU must hold for that length.

The chosen files are copied as rung_00.fasta ... rung_NN.fasta (zero-padded,
shortest = rung_00). That naming matters downstream:

  * workflow/scripts/chunk_fastas.py --max_files_per_job 1 slices the input in
    *sorted-filename* order, so zero-padded names make chunk_id == rung_idx.
  * workflow/scripts/organize_msa_outputs.py names each per-seq dir / YAML by
    the file stem (-> rung_NN), while the original FASTA *header* still supplies
    the Boltz chain protein_id. We therefore preserve the original header.

Usage:
    python scripts/calibrate/stair/pick_rungs.py \
        --input_dir /Users/thomasbush/tmp-data/tmp_data/GA_data/ga_fasta \
        --output_dir calib_stair_h100/staircase \
        --min 200 --max 3000 --step 200

Writes:
    output_dir/rung_00.fasta ... rung_NN.fasta   copies (original header kept)
    output_dir/rungs.csv                         rung_idx,target_len,actual_len,
                                                 accession,source_path
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

# Reuse the FASTA length parser from the sibling calibrate module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subsample import parse_fasta_length  # noqa: E402


def parse_accession(path: Path) -> str:
    """First header token, '>' stripped, before any '|' — matches the protein_id
    that organize_msa_outputs.get_protein_id() would assign."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    tok = line.lstrip(">").split()[0] if line.lstrip(">").split() else ""
                    return tok.split("|")[0] or path.stem
    except OSError:
        pass
    return path.stem


def pick_nearest(
    paths_with_len: list[tuple[Path, int]],
    targets: list[int],
) -> list[tuple[int, int, Path, int]]:
    """Greedy nearest-target assignment with no reuse.

    Returns list of (target_len, rung_idx, path, actual_len) in target order.
    Targets are processed ascending; each takes the closest still-unused
    sequence. Ties break by (abs diff, length, name) for determinism.
    """
    used: set[Path] = set()
    out: list[tuple[int, int, Path, int]] = []
    for rung_idx, target in enumerate(sorted(targets)):
        candidates = [(p, L) for p, L in paths_with_len if p not in used]
        if not candidates:
            raise SystemExit(
                f"ERROR: ran out of unused sequences at target {target} "
                f"(need {len(targets)} distinct, have {len(paths_with_len)})"
            )
        p, L = min(candidates, key=lambda x: (abs(x[1] - target), x[1], x[0].name))
        used.add(p)
        out.append((target, rung_idx, p, L))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input_dir", type=Path, required=True,
                    help="Directory of real .fasta/.fa files to draw rungs from")
    ap.add_argument("--output_dir", type=Path, required=True,
                    help="Where rung_NN.fasta + rungs.csv are written")
    ap.add_argument("--min", type=int, default=200, help="First target length")
    ap.add_argument("--max", type=int, default=3000, help="Last target length (inclusive)")
    ap.add_argument("--step", type=int, default=200, help="Length step between rungs")
    ap.add_argument("--mode", choices=["copy", "symlink"], default="copy",
                    help="copy is portable across mounts; symlink is faster (default: copy)")
    args = ap.parse_args()

    if args.min <= 0 or args.step <= 0 or args.max < args.min:
        print("ERROR: require 0 < min <= max and step > 0", file=sys.stderr)
        return 1

    targets = list(range(args.min, args.max + 1, args.step))
    print(f"Targets ({len(targets)} rungs): {targets}")

    fastas = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix in (".fasta", ".fa")
    )
    if not fastas:
        print(f"ERROR: no .fasta/.fa in {args.input_dir}", file=sys.stderr)
        return 1

    print(f"Scanning {len(fastas)} files for length...")
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
    if len(paths_with_len) < len(targets):
        print(f"ERROR: only {len(paths_with_len)} usable seqs for {len(targets)} rungs",
              file=sys.stderr)
        return 1

    picked = pick_nearest(paths_with_len, targets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Clean stale rung_*.fasta so reruns don't leave orphans.
    for stale in args.output_dir.glob("rung_*.fasta"):
        stale.unlink()

    rungs_csv = args.output_dir / "rungs.csv"
    # lineterminator="\n" (not csv's default "\r\n") so the bash bench wrapper
    # can read rungs.csv without trailing carriage returns mangling paths.
    with rungs_csv.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["rung_idx", "target_len", "actual_len", "accession", "source_path"])
        for target, rung_idx, src, L in picked:
            dest = args.output_dir / f"rung_{rung_idx:02d}.fasta"
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            if args.mode == "symlink":
                dest.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dest)  # preserves original header
            w.writerow([rung_idx, target, L, parse_accession(src), str(src.resolve())])

    print(f"\nWrote {len(picked)} rungs to {args.output_dir}")
    print(f"Manifest: {rungs_csv}\n")
    print(f"  {'rung':>4}  {'target':>6}  {'actual':>6}  {'delta':>6}  accession")
    worst = 0
    for target, rung_idx, src, L in picked:
        d = L - target
        worst = max(worst, abs(d))
        print(f"  {rung_idx:>4}  {target:>6}  {L:>6}  {d:>+6}  {parse_accession(src)}")
    print(f"\nLargest deviation from target: {worst} aa")
    return 0


if __name__ == "__main__":
    sys.exit(main())
