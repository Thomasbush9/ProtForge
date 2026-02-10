import os
import re
import shutil
from argparse import ArgumentParser
from ast import parse
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from .utils import balanced_sampling, sample_with_num_mutations


def _infer_sep(path: str) -> str:
    return "," if path.lower().endswith(".csv") else "\t"


if __name__ == "__main__":
    print("Generating subsample")

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument(
        "--main_dir", type=str, required=True, help="Directory containing files"
    )
    parser.add_argument("--mode", type=str, choices=["balanced", "fixed"], default="balanced")
    parser.add_argument("--num_mut", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sep", type=str, default=None,
        help="Column separator (default: ',' for .csv, else tab)",
    )

    timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_subsample"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", timestamp)
    os.makedirs(data_dir, exist_ok=True)
    sub_dir = os.path.join(data_dir, "subsample")
    os.makedirs(sub_dir, exist_ok=True)

    args = parser.parse_args()
    data_path = Path(args.dataset)
    n = args.n
    main_dir = Path(args.main_dir)
    mode = args.mode
    seed = args.seed
    sep = args.sep if args.sep is not None else _infer_sep(str(data_path))

    # Load the dataset
    print(f"Loading dataset from: {data_path}")
    dataset = pd.read_csv(data_path, sep=sep)

    # Generate balanced subsample and create txt file
    if mode == "balanced":
        output_txt_path = os.path.join(data_dir, "balanced_subset.txt")
        balanced_dataset = balanced_sampling(dataset, n, output_txt_path, seed)
    elif mode == "fixed":
        num_muts = args.num_mut
        output_txt_path = os.path.join(data_dir, f"n{num_muts}_subset.txt")
        balanced_dataset = sample_with_num_mutations(dataset, n, num_muts, output_txt_path, seed)

    # Copy corresponding YAML files
    print(f"Copying the files from: {main_dir}")
    copied_count = 0

    # Determine the maximum index to calculate padding length
    max_idx = max(balanced_dataset.index)
    padding_length = len(str(max_idx))

    # Determine the type of files:
    file_1 = [f for f in Path(main_dir).iterdir()][0]

    for idx in balanced_dataset.index:
        # Format index with zero padding
        padded_idx = str(idx).zfill(padding_length)
        filename = f"seq_{padded_idx}" + file_1.suffix
        source_path = main_dir / filename
        dest_path = Path(sub_dir) / filename

        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            copied_count += 1
        else:
            print(f"Warning: file not found: {source_path}")

    print(f"Successfully copied {copied_count}  files to: {data_dir}")
    print(f"Total files in output directory: {len(list(Path(data_dir).glob('*')))}")
    print(f"SUBSAMPLE_OUTPUT_DIR={os.path.abspath(sub_dir)}")
