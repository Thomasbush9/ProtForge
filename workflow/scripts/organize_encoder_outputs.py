#!/usr/bin/env python3
"""
Relocate batch-encoder outputs into the canonical sequences/ tree.

The in-container batch runners (containers/run_batch_esmc.py,
containers/run_batch_esmfold.py) write to a scratch directory laid out as:

    {scratch}/{name}/{sub}/...

where {name} is the input YAML stem and {sub} is the ESMC model size
(6B/600M/300M) or the ESMFold2 variant (fast). This script moves each
{name}/{sub} payload to:

    {sequences_dir}/{name}/{stage}/{sub}/...

so encoder outputs sit beside msa/ and boltz/ under each sequence. It is
stage-agnostic: pass --stage esmc or --stage esmfold. Reruns overwrite any
previously organized {sub} for the same sequence.
"""

import argparse
import shutil
import sys
from pathlib import Path


def organize(scratch_dir: str, sequences_dir: str, stage: str) -> int:
    """Move every {name}/{sub} under scratch into sequences/{name}/{stage}/{sub}."""
    scratch = Path(scratch_dir)
    seq_base = Path(sequences_dir)

    moved = 0
    for name_dir in sorted(p for p in scratch.iterdir() if p.is_dir()):
        dest = seq_base / name_dir.name / stage
        dest.mkdir(parents=True, exist_ok=True)
        for sub in sorted(name_dir.iterdir()):
            target = dest / sub.name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            shutil.move(str(sub), target)
            moved += 1
            print(f"Organized: {name_dir.name}/{sub.name} -> {target}")
    return moved


def main():
    parser = argparse.ArgumentParser(description="Move encoder outputs into sequences/")
    parser.add_argument("--scratch_dir", required=True, help="Per-chunk scratch output dir")
    parser.add_argument("--sequences_dir", required=True, help="Base sequences directory")
    parser.add_argument("--stage", required=True, choices=["esmc", "esmfold", "sae"],
                        help="Target subdirectory under each sequence")
    args = parser.parse_args()

    moved = organize(args.scratch_dir, args.sequences_dir, args.stage)
    if moved == 0:
        print(f"ERROR: no outputs found under {args.scratch_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Moved {moved} output dir(s) into {args.sequences_dir}")


if __name__ == "__main__":
    main()
