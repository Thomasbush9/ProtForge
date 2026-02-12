import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from .utils import (
    generate_cluster_fasta,
    generate_fasta_data,
    generate_fasta_from_sequences,
    generate_yaml_data,
    generate_yaml_from_sequences,
    load_dataset,
    load_seq_,
    mutate_sequence,
)

def _infer_sep(path: str) -> str:
    """Infer delimiter from file extension: .csv -> ',', else tab."""
    return "," if path.lower().endswith(".csv") else "\t"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pipeline input files from mutations or sequence CSV"
    )
    parser.add_argument("--data", type=str, required=True,
        help="CSV/TSV file. For mutations mode: must have 'aaMutations' column. "
             "For sequences mode: must have 'name' and 'sequence' columns.")
    parser.add_argument("--msa", type=str, default=None,
        help="MSA file path (stored in output files)")
    parser.add_argument("--original", type=str, default=None,
        help="Reference sequence (.fasta or .yaml). Required for mutations mode.")
    parser.add_argument(
        "--file_type", type=str, choices=["cluster", "fasta", "yaml"], default="fasta",
        help="Output format: cluster (FASTA for MSA), fasta, or yaml (skip MSA)"
    )
    parser.add_argument(
        "--mode", type=str, choices=["mutations", "sequences"], default=None,
        help="Input mode: 'mutations' (requires --original) or 'sequences' (name/sequence CSV). "
             "Auto-detected if not specified."
    )
    parser.add_argument(
        "--sep", type=str, default=None,
        help="Column separator (default: ',' for .csv, else tab)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: data/<timestamp>)",
    )
    args = parser.parse_args()

    msa = args.msa if args.msa else "empty"
    dataset_path = args.data
    file_type = args.file_type
    sep = args.sep if args.sep is not None else _infer_sep(dataset_path)

    # Load dataset to detect mode
    dataset = load_dataset(dataset_path, sep=sep)

    # Auto-detect or validate mode
    has_mutations = "aaMutations" in dataset.columns
    has_sequences = "name" in dataset.columns and "sequence" in dataset.columns

    if args.mode:
        input_mode = args.mode
    elif has_sequences and not has_mutations:
        input_mode = "sequences"
        print("[INFO] Auto-detected sequences mode (found 'name' and 'sequence' columns)")
    elif has_mutations:
        input_mode = "mutations"
        print("[INFO] Auto-detected mutations mode (found 'aaMutations' column)")
    else:
        raise ValueError(
            "Could not detect input mode. CSV must have either:\n"
            "  - 'aaMutations' column (mutations mode, also requires --original)\n"
            "  - 'name' and 'sequence' columns (sequences mode)"
        )

    # Setup output directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if args.output_dir:
        data_dir = os.path.abspath(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = os.path.join(base_dir, "data", timestamp)
    training_data_dir = os.path.join(data_dir, "training_data")
    os.makedirs(training_data_dir, exist_ok=True)

    if input_mode == "mutations":
        # Mutations mode: requires original sequence
        if not args.original:
            raise ValueError("Mutations mode requires --original (reference sequence)")

        original_seq_path = args.original
        if original_seq_path.endswith(".yaml"):
            seq, mapping = load_seq_(original_seq_path, fasta=False)
        else:
            seq, mapping = load_seq_(original_seq_path)

        dataset["seq_mutated"] = dataset["aaMutations"].apply(
            lambda muts: mutate_sequence(muts, seq=seq, mapping_db_seq=mapping)
        )

        if file_type == "yaml":
            generate_yaml_data(dataset, msa, training_data_dir, data_dir)
        elif file_type == "fasta":
            generate_fasta_data(dataset, msa, training_data_dir, data_dir)
        else:  # cluster
            generate_cluster_fasta(dataset, training_data_dir, data_dir)

    else:  # sequences mode
        if file_type == "yaml":
            generate_yaml_from_sequences(dataset, training_data_dir, data_dir, msa)
        else:  # fasta or cluster - both output FASTA
            generate_fasta_from_sequences(dataset, training_data_dir, data_dir)

    print("")
    print("All files have been generated correctly.")
    print(f"Output directory: {training_data_dir}")
