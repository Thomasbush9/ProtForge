"""Generate one FASTA per single-point mutant of a wild-type sequence.

The Saturation Mutagenesis tab scores every substitution from a *single* ESM-C
forward pass on the wild type (a wt-marginal log-likelihood ratio), so it never
embeds the mutants themselves. When you want a real per-mutant embedding — or a
structure — you need the mutant sequences as actual pipeline input. That is what
this produces: a directory of FASTA files that `input.fasta_dir` can point at.

    python utils/generate_satmut.py --input wt.fasta --output-dir muts/

For a length-L sequence over the 20 standard amino acids that is L x 19 files
(the wild-type residue at each position is skipped). A 238-residue protein gives
4522 sequences — a normal-sized ProtForge run, but check the count before
launching a stage over it. A selenocysteine position contributes 20, not 19,
since none of the 20 substitutions is its wild type.

Names follow the standard point-mutation convention `<wt><pos><mut>` with
1-based positions (`M1A`, `K52R`), optionally prefixed with the parent name so
several scans can share one directory. `index.csv` records the mapping from
file name to mutation and sequence, for joining predictions back to variants.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.utils import AMINO_ACIDS_20, load_seq_

# Wild-type residues we can meaningfully mutate *away from*. Selenocysteine is a
# real residue in selenoproteins (the deiodinases, the glutathione peroxidases)
# and is in the ESM-C vocabulary, so its position is scannable — often it is the
# catalytic one. It stays out of the substitution alphabet because Sec insertion
# requires a SECIS element, so introducing it elsewhere is not meaningful.
MUTABLE_WT_AAS = set(AMINO_ACIDS_20) | {"U"}


def parse_positions(spec: str | None, length: int) -> list[int]:
    """Parse a 1-based position selector like "1-50,73,80-90" into indices.

    None means every position. Raises ValueError on a range outside the
    sequence, rather than silently scanning fewer positions than asked for.
    """
    if spec is None:
        return list(range(1, length + 1))

    positions: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, _, hi_s = chunk.partition("-")
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"range '{chunk}' runs backwards")
            positions.update(range(lo, hi + 1))
        else:
            positions.add(int(chunk))

    out_of_range = sorted(p for p in positions if p < 1 or p > length)
    if out_of_range:
        raise ValueError(
            f"positions {out_of_range} are outside the sequence (length {length})"
        )
    return sorted(positions)


def satmut_variants(
    wt_seq: str,
    positions: list[int],
    alphabet: str = "".join(AMINO_ACIDS_20),
    prefix: str = "",
) -> list[tuple[str, str, str]]:
    """Every single substitution at `positions`, as (name, mutation, sequence).

    The wild-type residue is skipped at each position, so a position
    contributes len(alphabet) - 1 variants. Positions are 1-based.
    """
    variants: list[tuple[str, str, str]] = []
    for pos in positions:
        wt_aa = wt_seq[pos - 1]
        for mut_aa in alphabet:
            if mut_aa == wt_aa:
                continue
            mutation = f"{wt_aa}{pos}{mut_aa}"
            seq = wt_seq[: pos - 1] + mut_aa + wt_seq[pos:]
            variants.append((f"{prefix}{mutation}", mutation, seq))
    return variants


def split_scannable(seq: str, positions: list[int]) -> tuple[list[int], list[int]]:
    """Partition `positions` into (scannable, skipped) by wild-type residue.

    Ambiguity codes (X, B, Z) and gaps can't be substituted meaningfully, so
    they are dropped rather than turned into nonsense variants. Everything in
    MUTABLE_WT_AAS is kept, selenocysteine included.
    """
    keep = [p for p in positions if seq[p - 1] in MUTABLE_WT_AAS]
    skip = [p for p in positions if seq[p - 1] not in MUTABLE_WT_AAS]
    return keep, skip


def write_fastas(
    variants: list[tuple[str, str, str]], out_dir: Path, line_width: int = 60
) -> None:
    """Write one single-record FASTA per variant, wrapped at `line_width`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, _, seq in variants:
        wrapped = "\n".join(
            seq[i : i + line_width] for i in range(0, len(seq), line_width)
        )
        (out_dir / f"{name}.fasta").write_text(f">{name}\n{wrapped}\n")


def write_index(variants: list[tuple[str, str, str]], path: Path) -> None:
    """Record name -> mutation -> sequence, for joining results back later."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", "mutation", "position", "wt_aa", "mut_aa", "sequence"])
        for name, mutation, seq in variants:
            wt_aa, mut_aa = mutation[0], mutation[-1]
            writer.writerow([name, mutation, mutation[1:-1], wt_aa, mut_aa, seq])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True,
                        help="Wild-type sequence (.fasta/.a3m or .yaml), single entry.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for the mutant FASTAs; created if absent.")
    parser.add_argument("--alphabet", default="".join(AMINO_ACIDS_20),
                        help="Substitutions to try. Default: the 20 standard AAs.")
    parser.add_argument("--positions", default=None,
                        help="1-based subset, e.g. '1-50,73'. Default: every position.")
    parser.add_argument("--name-prefix", default="",
                        help="Prepended to each file name, e.g. 'GFP_' -> GFP_M1A.fasta.")
    parser.add_argument("--include-wt", action="store_true",
                        help="Also emit the unmutated sequence as <prefix>WT.fasta, "
                             "so a run has its own baseline to compare against.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report how many files would be written, and stop.")
    args = parser.parse_args()

    seq, _ = load_seq_(args.input, fasta=Path(args.input).suffix != ".yaml")
    seq = str(seq).strip().upper()
    if not seq:
        raise SystemExit(f"ERROR: no sequence found in {args.input}")

    alphabet = args.alphabet.strip().upper()
    unknown = sorted(set(alphabet) - set(AMINO_ACIDS_20))
    if unknown:
        raise SystemExit(f"ERROR: --alphabet contains non-standard residues: {unknown}")

    try:
        positions = parse_positions(args.positions, len(seq))
    except ValueError as exc:
        raise SystemExit(f"ERROR: --positions {exc}")

    positions, skipped = split_scannable(seq, positions)

    variants = satmut_variants(seq, positions, alphabet, args.name_prefix)
    if args.include_wt:
        variants.append((f"{args.name_prefix}WT", "WT", seq))

    print(f"sequence length : {len(seq)}")
    print(f"positions       : {len(positions)}")
    print(f"alphabet        : {len(alphabet)} residues")
    print(f"files to write  : {len(variants)}")
    if skipped:
        shown = ", ".join(f"{seq[p - 1]}{p}" for p in skipped[:10])
        if len(skipped) > 10:
            shown += f", +{len(skipped) - 10} more"
        print(f"skipped         : {len(skipped)} position(s) with non-standard "
              f"residues ({shown})")
    if args.dry_run:
        print("(--dry-run: nothing written)")
        return

    out_dir = Path(args.output_dir)
    write_fastas(variants, out_dir)
    write_index(variants, out_dir / "index.csv")
    print(f"\nWrote {len(variants)} FASTA files to {out_dir}")
    print(f"Index: {out_dir / 'index.csv'}")
    print(f"\nPoint input.fasta_dir at {out_dir} to run the pipeline over these.")


if __name__ == "__main__":
    main()
