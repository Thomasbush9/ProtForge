#!/usr/bin/env python3
"""
Organize Boltz prediction outputs: pick the highest-confidence model per sequence
and copy it to the canonical sequences/{seq_name}/boltz/ directory.

Extracted from: slurm_scripts/organize_boltz_outputs.sh
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


def find_predictions_dir(boltz_output_dir: str, chunk_id: str) -> Path | None:
    """Find the predictions directory within boltz output for a given chunk."""
    output_path = Path(boltz_output_dir)
    # Pattern: boltz_results_chunk_N/predictions/
    for results_dir in output_path.glob(f"boltz_results_chunk_{chunk_id}"):
        pred_dir = results_dir / "predictions"
        if pred_dir.is_dir():
            return pred_dir
    # Fallback: try any boltz_results_*/predictions/
    for results_dir in output_path.glob("boltz_results_*"):
        pred_dir = results_dir / "predictions"
        if pred_dir.is_dir():
            return pred_dir
    return None


def organize_chunk(boltz_output_dir: str, chunk_id: str, sequences_dir: str,
                   delete_msa: bool = False):
    """
    For one chunk's boltz output, find the highest model number per sequence
    and copy those files to sequences/{seq_name}/boltz/.
    """
    predictions_dir = find_predictions_dir(boltz_output_dir, chunk_id)
    if predictions_dir is None:
        print(f"ERROR: No predictions directory found in {boltz_output_dir}")
        sys.exit(1)

    seq_base = Path(sequences_dir)
    model_pattern = re.compile(r"_model_(\d+)\.")

    processed = 0
    skipped = 0

    for pred_subdir in sorted(predictions_dir.iterdir()):
        if not pred_subdir.is_dir():
            continue

        # Get sequence ID from directory name (may have seq_ prefix)
        dir_name = pred_subdir.name
        seq_id = dir_name.removeprefix("seq_") if dir_name.startswith("seq_") else dir_name

        # Find highest model number among files in this prediction dir
        highest_model = -1
        model_files: dict[int, list[Path]] = {}

        for f in pred_subdir.iterdir():
            if not f.is_file():
                continue
            match = model_pattern.search(f.name)
            if match:
                model_num = int(match.group(1))
                model_files.setdefault(model_num, []).append(f)
                if model_num > highest_model:
                    highest_model = model_num

        if highest_model < 0:
            print(f"WARNING: No model files found for {dir_name}")
            skipped += 1
            continue

        # Determine target dir: look for existing sequence dir matching seq_id
        # Try both seq_{id} and plain {id} formats
        target_dir = None
        for candidate in [f"seq_{seq_id}", seq_id]:
            candidate_dir = seq_base / candidate / "boltz"
            # Check if the parent exists (sequence dir from MSA stage)
            if (seq_base / candidate).exists():
                target_dir = candidate_dir
                break
        if target_dir is None:
            # Default to seq_{id} format
            target_dir = seq_base / f"seq_{seq_id}" / "boltz"

        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy highest model files
        for f in model_files[highest_model]:
            shutil.copy2(f, target_dir / f.name)

        print(f"Organized {dir_name}: model {highest_model} -> {target_dir}")
        processed += 1

        # Optionally delete MSA directory
        if delete_msa:
            msa_dir = target_dir.parent / "msa"
            if msa_dir.is_dir():
                shutil.rmtree(msa_dir)
                print(f"  Deleted MSA directory for {dir_name}")

    print(f"Processed: {processed}, Skipped: {skipped}")
    return processed


def main():
    parser = argparse.ArgumentParser(description="Organize Boltz prediction outputs")
    parser.add_argument("--boltz_output_dir", required=True,
                        help="Boltz output directory for a chunk (e.g. boltz_chunks/chunk_0_output)")
    parser.add_argument("--chunk_id", required=True, help="Chunk ID (e.g. 0)")
    parser.add_argument("--sequences_dir", required=True, help="Base sequences directory")
    parser.add_argument("--delete_msa", action="store_true",
                        help="Delete MSA directories after organizing")
    args = parser.parse_args()

    processed = organize_chunk(
        args.boltz_output_dir, args.chunk_id, args.sequences_dir, args.delete_msa
    )
    if processed == 0:
        print("ERROR: No predictions were organized", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
