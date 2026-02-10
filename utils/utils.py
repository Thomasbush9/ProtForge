import os
import re
from pathlib import Path
from typing import Literal, Optional
import random
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

AMINO_ACIDS_20 = [
    "A",  # Alanine
    "R",  # Arginine
    "N",  # Asparagine
    "D",  # Aspartic acid
    "C",  # Cysteine
    "E",  # Glutamic acid
    "Q",  # Glutamine
    "G",  # Glycine
    "H",  # Histidine
    "I",  # Isoleucine
    "L",  # Leucine
    "K",  # Lysine
    "M",  # Methionine
    "F",  # Phenylalanine
    "P",  # Proline
    "S",  # Serine
    "T",  # Threonine
    "W",  # Tryptophan
    "Y",  # Tyrosine
    "V",  # Valine
]


def load_dataset(path: str, sep: None) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)


# def load_seq_(path: str):
#     with open(path, "r") as f:
#         file = f.readlines()
#         seq = "".join([s.strip("\n") for s in file[1:]])
#         mapping_db_seq = {str(i): i + 1 for i in range(len(seq))}
#         return seq, mapping_db_seq
def load_seq_(path: str, return_meta: bool = False, fasta: bool = True):
    """
    Load a single-entry FASTA/A3M or YAML protein file.

    Parameters
    ----------
    path : str
        Path to the FASTA/A3M or YAML file.
    return_meta : bool, optional
        If True, also return a dict with 'type', 'idx', 'path', and 'header' (for FASTA).

    Returns
    -------
    seq : str
        The sequence from the file.
    mapping_db_seq : dict
        Maps sequence position (as str) to 1-based index.
    meta : dict, optional
        Returned only if return_meta=True and input is FASTA/A3M.
    """
    open_args = {"mode": "r"}
    if not fasta:
        # YAML: open, load, extract sequence
        with open(path, **open_args) as f:
            data = yaml.safe_load(f)
        
        if data is None:
            raise ValueError(f"YAML file {path} is empty or could not be parsed")
        
        if "sequences" not in data:
            available_keys = list(data.keys()) if isinstance(data, dict) else "not a dict"
            raise KeyError(
                f"YAML file {path} does not contain 'sequences' key. "
                f"Available keys: {available_keys}. "
                f"Full data structure: {data}"
            )
        
        if not isinstance(data["sequences"], list) or len(data["sequences"]) == 0:
            raise ValueError(
                f"YAML file {path} has 'sequences' but it is not a non-empty list. "
                f"Value: {data['sequences']}"
            )
        
        if "protein" not in data["sequences"][0]:
            raise KeyError(
                f"YAML file {path} sequence entry does not contain 'protein' key. "
                f"Available keys: {list(data['sequences'][0].keys())}"
            )
        
        if "sequence" not in data["sequences"][0]["protein"]:
            raise KeyError(
                f"YAML file {path} protein entry does not contain 'sequence' key. "
                f"Available keys: {list(data['sequences'][0]['protein'].keys())}"
            )
        
        seq = data["sequences"][0]["protein"]["sequence"]
        mapping_db_seq = {str(i): i + 1 for i in range(len(seq))}
        # For YAML no meta info is available/meaningful for return_meta,
        # so just return two outputs regardless of return_meta
        return seq, mapping_db_seq
    else:
        # FASTA/A3M: open, parse header and sequence lines
        with open(path, **open_args) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines or not lines[0].startswith(">"):
            raise ValueError(
                f"File {path} does not look like a FASTA/A3M with a header on the first line."
            )
        header = lines[0]
        seq = "".join(lines[1:])
        mapping_db_seq = {str(i): i + 1 for i in range(len(seq))}
        if return_meta:
            header_body = header[1:]  # remove ">"
            parts = header_body.split("|", maxsplit=2)
            seq_type, idx, msa_path = (parts + [None, None, None])[:3]
            meta = {"type": seq_type, "idx": idx, "path": msa_path, "header": header}
            return seq, mapping_db_seq, meta
        return seq, mapping_db_seq


def parse_mutation(mutation_str: str):
    match = re.match(r"^S([A-Za-z])(\d+)(.+)$", mutation_str)
    if match:
        return match.groups()
    else:
        raise ValueError(f"Invalid mutation format: {mutation_str}")


def mutate_seq(
    src: str, idx: str, dest: str, seq: str, mapping: dict, mapping_db_seq
) -> str:
    idx = mapping_db_seq[idx]
    new_seq = seq[:idx] + dest + seq[idx + 1 :]
    return new_seq


def mutate_sequence(mutation_string, seq, mapping_db_seq):
    if pd.isna(mutation_string):
        return None
    try:
        mutations = mutation_string.split(":")
        for m in mutations:
            src, idx, dest = parse_mutation(m)
            if idx in mapping_db_seq:
                mapped_idx = mapping_db_seq[idx]
                mutated_seq = seq[:mapped_idx] + dest + seq[mapped_idx + 1 :]
                return mutated_seq
    except Exception:
        return None
    return None

    # Calculate the original distribution of num_mut
    def get_numb_mut(mut: str) -> int:
        if type(mut) == str:
            n = len(mut.split(":"))
        else:
            return 0
        return n

def get_numb_mut(mut: str) -> int:
    """Helper function to count number of mutations from mutation string"""
    if type(mut) == str:
        n = len(mut.split(":"))
    else:
        return 0
    return n
def sample_with_num_mutations(
    dataset: pd.DataFrame,
    n: int,
    num_mutations: int,
    output_file: Optional[str] = None,
    seed: Optional[int] = None,
    replace: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Sample rows that have exactly `num_mutations` mutations.

    Parameters
    ----------
    dataset : pd.DataFrame
        Must contain a column 'aaMutations'. A helper `get_numb_mut` must exist.
    n : int
        Number of rows to sample.
    num_mutations : int
        The exact mutation count to filter by.
    output_file : Optional[str]
        If provided, writes a tab-separated file with (idx, n_mut).
    seed : Optional[int]
        RNG seed for reproducibility (uses numpy Generator).
    replace : bool
        If True, allows sampling with replacement.
    strict : bool
        If True and there are fewer than `n` eligible rows (and replace=False),
        raise a ValueError. If False, will downshift `n` to the available count.

    Returns
    -------
    pd.DataFrame
        The sampled subset (copy).
    """
    rng = np.random.default_rng(seed)

    # Work on a copy to avoid mutating the original df
    df = dataset.copy()
    df["num_mut"] = df["aaMutations"].apply(lambda mut: get_numb_mut(mut))

    # Filter to the desired mutation count
    eligible = df[df["num_mut"] == num_mutations]

    if eligible.empty:
        raise ValueError(
            f"No rows found with num_mutations == {num_mutations} "
            f"(eligible size = 0)."
        )

    if not replace:
        if n > len(eligible):
            if strict:
                raise ValueError(
                    f"Requested n ({n}) exceeds available rows with "
                    f"{num_mutations} mutations ({len(eligible)})."
                )
            # downshift n to what we have
            n = len(eligible)

        # sample without replacement using numpy for reproducibility
        indices = eligible.index.to_numpy()
        chosen = rng.choice(indices, size=n, replace=False)
    else:
        # with replacement
        indices = eligible.index.to_numpy()
        chosen = rng.choice(indices, size=n, replace=True)

    # Optional: stable order in output (sorted by original index)
    chosen = np.sort(chosen)

    subset = df.loc[chosen].copy()

    if output_file:
        with open(output_file, "w") as f:
            f.write("idx\tn_mut\n")
            for idx in chosen:
                f.write(f"{idx}\t{df.loc[idx, 'num_mut']}\n")

        print(f"Subset created with {len(subset)} samples (num_mut={num_mutations}).")
        print(f"Results saved to: {output_file}")

    return subset

def balanced_sampling(
    dataset: pd.DataFrame,
    n: int,
    output_file="balanced_subset.txt",
    seed: Optional[int] = None,
):
    if n >= len(dataset):
        raise ValueError(f"n ({n}) must be less than dataset size ({len(dataset)})")

    rng = np.random.default_rng(seed)

    # Work on a copy to avoid mutating the original df
    df = dataset.copy()
    df["num_mut"] = df["aaMutations"].apply(lambda mut: get_numb_mut(mut))
    original_dist = df["num_mut"].value_counts(normalize=True)

    target_counts = (original_dist * n).round().astype(int)

    available_counts = df["num_mut"].value_counts()
    final_counts = {
        k: min(target_counts.get(k, 0), available_counts.get(k, 0))
        for k in target_counts.index
    }

    total_selected = sum(final_counts.values())
    if total_selected < n:
        remaining = n - total_selected
        for num_mut in sorted(
            original_dist.index, key=lambda x: original_dist[x], reverse=True
        ):
            if remaining <= 0:
                break
            available = available_counts.get(num_mut, 0) - final_counts.get(num_mut, 0)
            to_add = min(remaining, max(available, 0))
            if to_add > 0:
                final_counts[num_mut] = final_counts.get(num_mut, 0) + to_add
                remaining -= to_add

    balanced_subset = []
    for num_mut, count in final_counts.items():
        if count > 0:
            category_indices = df.index[df["num_mut"] == num_mut].to_numpy()
            chosen = rng.choice(category_indices, size=count, replace=False)
            balanced_subset.extend(chosen.tolist())

    # Optional: stable order in output file
    balanced_subset = sorted(balanced_subset)

    balanced_dataset = df.loc[balanced_subset].copy()

    with open(output_file, "w") as f:
        f.write("idx\tn_mut\n")
        for idx in balanced_subset:
            f.write(f"{idx}\t{df.loc[idx, 'num_mut']}\n")

    print(f"Balanced subset created with {len(balanced_dataset)} samples")
    print("Original distribution:")
    print(df["num_mut"].value_counts(normalize=True).sort_index())
    print("\nBalanced subset distribution:")
    print(balanced_dataset["num_mut"].value_counts(normalize=True).sort_index())
    print(f"\nResults saved to: {output_file}")

    return balanced_dataset



# === Generate Data Functions ===
def generate_yaml_data(dataset: pd.DataFrame, msa, training_data_dir, data_dir):
    """
    Function to generate yaml data for Boltz Predictions.

    Args:
    - dataset: dataset with the mutated sequences, it must have the following
      columns: seq_mutated
    - msa: path to the msa path, if None it will be skipped
    - training_data_dir: where to save the mutated sequences
    - data_dir: where to save the idx

    It saves each mutated sequences in the yaml format in its own separate folder with this structure:
    data_dir
        index.csv
        training_data
            seq_00001
                seq_0001.yaml
    """
    index_records = []
    for idx, row in tqdm(dataset.iterrows(), desc="Generating data"):
        mutated_seq = row["seq_mutated"]

        data_seq = {
            "version": 1,
            "sequences": [
                {"protein": {"id": str(idx), "sequence": mutated_seq, "msa": msa}}
            ],
        }

        filename = f"seq_{idx:05}.yaml"
        filepath = os.path.join(training_data_dir, filename)

        try:
            with open(filepath, "w") as file:
                yaml.dump(data_seq, file, sort_keys=False)
            index_records.append({"idx": idx, "filename": filename})
        except Exception as e:
            print(f"[✗] Failed to write {filename}: {e}")
    # 4. Save index.csv
    index_df = pd.DataFrame(index_records)
    index_df.to_csv(os.path.join(data_dir, "index.csv"), index=False)
    print(f"[✓] Index file written to: {os.path.join(data_dir, 'index.csv')}")


def generate_fasta_data(dataset: pd.DataFrame, msa, training_data_dir, data_dir):
    """
    Function to generate fasta data for Boltz Predictions.

    Args:
    - dataset: dataset with the mutated sequences, it must have the following
      columns: seq_mutated
    - msa: path to the msa path, if None it will be skipped
    - training_data_dir: where to save the mutated sequences
    - data_dir: where to save the idx

    It saves each mutated sequences in the fasta format in its own separate folder with this structure:
    data_dir
        index.csv
        training_data
            seq_00001
                seq_0001.fasta.txt
    """
    index_records = []

    for idx, row in tqdm(dataset.iterrows(), desc="Generating data"):
        mutated_seq = row["seq_mutated"]
        header = f">A|{idx}|{msa}"
        filename = f"seq_{idx:05}.fasta.txt"
        filepath = os.path.join(training_data_dir, filename)

        try:
            with open(filepath, "w") as f:
                f.write(header + "\n")
                f.write(mutated_seq + "\n")
            index_records.append({"idx": idx, "filename": filename})
        except Exception as e:
            print(f"[x] Failed to write {filename}: {e}")
    index_df = pd.DataFrame(index_records)
    index_df.to_csv(os.path.join(data_dir, "index.csv"), index=False)
    print(f"[✓] Index file written to: {os.path.join(data_dir, 'index.csv')}")


def generate_cluster_fasta(
    dataset: pd.DataFrame, training_data_dir: str, data_dir: str
):
    """
    Function to generate fasta data for Kempner Cluster Predictions.
    It works for the boltz workflow at: https://github.com/KempnerInstitute/boltz/tree/main/kempner_workflow/protein_fold_gpu

    Args:
    - dataset: dataset with the mutated sequences, it must have the following
      columns: seq_mutated
    - training_data_dir: where to save the mutated sequences
    - data_dir: where to save the idx

    It saves each mutated sequences in the fasta format in its own separate folder with this structure:
    data_dir
        index.csv
        training_data
            seq_00001
                seq_0001.fasta
    """
    index_records = []

    for idx, row in tqdm(dataset.iterrows(), desc="Generating data"):
        mutated_seq = row["seq_mutated"]
        header = f">{idx}|protein|"
        filename = f"seq_{idx:05}.fasta"
        filepath = os.path.join(training_data_dir, filename)

        try:
            with open(filepath, "w") as f:
                f.write(header + "\n")
                f.write(mutated_seq + "\n")
            index_records.append({"idx": idx, "filename": filename})
        except Exception as e:
            print(f"[x] Failed to write {filename}: {e}")
    index_df = pd.DataFrame(index_records)
    index_df.to_csv(os.path.join(data_dir, "index.csv"), index=False)
    print(f"[✓] Index file written to: {os.path.join(data_dir, 'index.csv')}")


# === easy converter ===
def fasta2yaml(path: str):
    """
    Function to convert .fasta.txt files in Boltx format to .yaml in Boltz format

    Args:
    - path: path to fasta files to convert
    """
    if os.path.isfile(path):
        seq, mapping, meta = load_seq_(path, return_meta=True)
        type_ = meta["type"]
        idx = meta["idx"]
        msa = meta["path"]
        header = meta["header"]

        old_suffix = ".fasta.txt"
        new_suffix = ".yaml"

        new_path = path.removesuffix(old_suffix) + new_suffix
        data_seq = {
            "version": 1,
            "sequences": [{"protein": {"id": idx, "sequence": seq, "msa": msa}}],
        }

        try:
            with open(new_path, "w") as file:
                yaml.dump(data_seq, file, sort_keys=False)
        except Exception as e:
            print(f"[✗] Failed to write {new_path}: {e}")


def yaml2fasta(path: str):
    """
    Load protein info from YAML and save as FASTA-like text file.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML file.
    output_path : str
        Path to save the output FASTA (.txt) file.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    old_suffix = ".yaml"
    new_suffix = ".fasta"
    output_path = path.removesuffix(old_suffix) + new_suffix

    # Expect structure: version, sequences -> list of protein dicts
    seq_entry = data["sequences"][0]["protein"]

    protein_id = seq_entry["id"]  # e.g., "A"
    msa_path = seq_entry["msa"]  # e.g., "raw_data/gfp_msa_b5fdc_0.a3m"
    sequence = seq_entry["sequence"]  # long amino acid string

    # Build header
    header = f">{protein_id}|protein|{msa_path}"

    # Write to output
    with open(output_path, "w") as out_f:
        out_f.write(header + "\n")
        out_f.write(sequence + "\n")

    print(f"[INFO] Saved FASTA file to {output_path}")


def converter(path: str, src: Literal["fasta", "yaml"]):
    """
    Converts a file or a directory from source to the other format

    Args:
    -path: [dir, file]
    - src: the source file that you want to covert, Literal['fasta', 'yaml']
    """
    # handle files:
    function = fasta2yaml if src == "fasta" else yaml2fasta
    suffix = "." + "yaml" if src == "yaml" else ".txt"
    if os.path.isfile(path):
        function(path)
        print(f"File {path} correctly converted")
    else:
        for file in tqdm(Path(path).iterdir()):
            if file.suffix == suffix:
                function(str(file))
            else:
                continue
        print(f"Directory {path} correctly converted")

#=== Functions to generate test data for GABRB3: 

def random_aa_except(exclude, amino_acids=AMINO_ACIDS_20):
    i = random.randrange(len(amino_acids) - 1)
    return amino_acids[i] if amino_acids[i] != exclude else amino_acids[-1]

def generate_mutation_dataset(seq:str, n:int)-> pd.DataFrame:

    mutations = []
    for i in range(0, n):
        idx = random.choice(range(0, len(seq)))
        src = seq[idx]
        aa = random_aa_except(src)
        mutations.append(f"S{src}{idx}{aa}")

    return pd.DataFrame({"aaMutations":mutations})







