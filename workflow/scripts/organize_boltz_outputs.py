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

import output_schema

_STRUCTURE_EXTS = (".cif", ".pdb", ".cif.gz")


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
                   delete_msa: bool = False, samples_to_save: "int | str" = 1,
                   subdirectory: str = ""):
    """
    For one chunk's boltz output, copy model files to sequences/{seq_name}/boltz/.

    Args:
        samples_to_save: "all" to copy every model file, or int N (>=1) to copy the
            top-N best models (model_0..model_(N-1); model_0 is best per Boltz).
        subdirectory: If set, outputs go to boltz/{subdirectory}/ (e.g. "run_0").
    """
    if samples_to_save != "all":
        samples_to_save = int(samples_to_save)
        if samples_to_save < 1:
            raise ValueError(f"samples_to_save must be >= 1 or 'all', got {samples_to_save}")

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

        # Find all model files grouped by model number
        model_files: dict[int, list[Path]] = {}

        for f in pred_subdir.iterdir():
            if not f.is_file():
                continue
            match = model_pattern.search(f.name)
            if match:
                model_num = int(match.group(1))
                model_files.setdefault(model_num, []).append(f)

        if not model_files:
            print(f"WARNING: No model files found for {dir_name}")
            skipped += 1
            continue

        # Determine target dir: look for existing sequence dir matching seq_id
        # Try both seq_{id} and plain {id} formats
        target_dir = None
        for candidate in [f"seq_{seq_id}", seq_id]:
            boltz_base = seq_base / candidate / "boltz"
            if subdirectory:
                candidate_dir = boltz_base / subdirectory
            else:
                candidate_dir = boltz_base
            # Check if the parent exists (sequence dir from MSA stage)
            if (seq_base / candidate).exists():
                target_dir = candidate_dir
                break
        if target_dir is None:
            boltz_base = seq_base / f"seq_{seq_id}" / "boltz"
            target_dir = boltz_base / subdirectory if subdirectory else boltz_base

        target_dir.mkdir(parents=True, exist_ok=True)

        # Reruns can change which model files should be kept. Remove previously
        # organized Boltz artifacts for this target so stale higher-numbered
        # models do not survive and confuse later CIF discovery.
        for existing in target_dir.iterdir():
            if existing.is_file() and (model_pattern.search(existing.name)
                                       or existing.name.endswith(output_schema.SUMMARY_SUFFIX)):
                existing.unlink()

        # Decide which model indices to keep. Boltz ranks model_0 as best, so
        # take the lowest-N indices for top-N selection.
        sorted_indices = sorted(model_files.keys())
        if samples_to_save == "all":
            keep_indices = sorted_indices
        else:
            keep_indices = sorted_indices[:samples_to_save]

        total_copied = 0
        for idx in keep_indices:
            for f in model_files[idx]:
                shutil.copy2(f, target_dir / f.name)
                total_copied += 1

        # Write a normalized summary sidecar per kept model (reads the confidence
        # JSON copied alongside the structure).
        for idx in keep_indices:
            structure = next(
                (f for f in model_files[idx]
                 if f.name.endswith(_STRUCTURE_EXTS)), None)
            if structure is not None:
                try:
                    output_schema.write_summary_for("boltz", target_dir / structure.name)
                except Exception as e:
                    print(f"  WARNING: could not write summary for {structure.name}: {e}")

        if samples_to_save == "all":
            label = f"all {len(keep_indices)} models"
        elif len(keep_indices) == 1:
            label = f"model {keep_indices[0]}"
        else:
            label = (f"top {len(keep_indices)} models "
                     f"(model_{keep_indices[0]}..model_{keep_indices[-1]})")
        print(f"Organized {dir_name}: {label} ({total_copied} files) -> {target_dir}")

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
    parser.add_argument("--samples_to_save", default="1",
                        help='Number of top models to keep (int >= 1) or "all". '
                             'Top-N picks model_0..model_(N-1) (model_0 is best per Boltz).')
    parser.add_argument("--subdirectory", default="",
                        help="Subdirectory under boltz/ (e.g. 'run_0' -> boltz/run_0/)")
    args = parser.parse_args()

    samples_to_save: "int | str"
    if args.samples_to_save == "all":
        samples_to_save = "all"
    else:
        try:
            samples_to_save = int(args.samples_to_save)
        except ValueError:
            parser.error(f"--samples_to_save must be a positive int or 'all', "
                         f"got {args.samples_to_save!r}")

    processed = organize_chunk(
        args.boltz_output_dir, args.chunk_id, args.sequences_dir,
        args.delete_msa, samples_to_save, args.subdirectory
    )
    if processed == 0:
        print("ERROR: No predictions were organized", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
