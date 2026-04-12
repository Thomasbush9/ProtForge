#!/usr/bin/env python3
"""
Organize colabfold_search outputs: scatter a3m files to per-sequence directories
and create Boltz-compatible YAML files.

For each sequence in a chunk's file_list.txt:
  1. Find the matching .a3m in colabfold_output/
  2. Copy it to sequences/{seq_name}/msa/
  3. Create sequences/{seq_name}/{seq_name}.yaml with protein ID, sequence, and MSA path

Extracted from: slurm_scripts/run_msa_array.slrm (a3m scattering) +
                slurm_scripts/process_msa_fasta.sh (FASTA->YAML conversion)
"""

import argparse
import shutil
import sys
from pathlib import Path


def parse_fasta_header(fasta_path: Path) -> tuple[str, str]:
    """Read a FASTA file and return (protein_prefix, sequence)."""
    with open(fasta_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"Not a valid FASTA file: {fasta_path}")
    header = lines[0]
    sequence = "".join(lines[1:])
    # Extract protein prefix: remove '>' and replace '|' with '_'
    protein_prefix = header.lstrip(">").split()[0].replace("|", "_")
    return protein_prefix, sequence


def get_protein_id(fasta_path: Path) -> str:
    """Extract protein ID from FASTA header for YAML output."""
    with open(fasta_path) as f:
        header = f.readline().strip()
    protein_id = header.lstrip(">").split()[0].split("|")[0]
    return protein_id if protein_id else fasta_path.stem


def organize_chunk(file_list: str, colabfold_output: str, sequences_dir: str):
    """
    For one chunk: match a3m files to original FASTAs, copy to per-seq dirs, create YAMLs.
    """
    file_list_path = Path(file_list)
    colabfold_path = Path(colabfold_output)
    seq_base = Path(sequences_dir)

    # Read the list of original FASTA paths
    fasta_paths = [
        Path(line.strip())
        for line in file_list_path.read_text().splitlines()
        if line.strip()
    ]

    # Collect all a3m files from colabfold output
    a3m_files = sorted(colabfold_path.glob("*.a3m"))
    # Build lookup by stem
    a3m_by_stem = {f.stem: f for f in a3m_files}

    processed = 0
    skipped = 0

    for i, fasta_path in enumerate(fasta_paths):
        if not fasta_path.exists():
            print(f"WARNING: FASTA not found: {fasta_path}")
            skipped += 1
            continue

        # Derive sequence name (filename without extension)
        seq_name = fasta_path.stem
        if seq_name.endswith(".fasta"):
            seq_name = seq_name[:-6]

        protein_prefix, sequence = parse_fasta_header(fasta_path)
        protein_id = get_protein_id(fasta_path)

        # Try to find matching a3m: by protein_prefix, then by seq_name, then by order
        a3m_file = None
        if protein_prefix in a3m_by_stem:
            a3m_file = a3m_by_stem[protein_prefix]
        elif seq_name in a3m_by_stem:
            a3m_file = a3m_by_stem[seq_name]
        else:
            # Try partial match
            for stem, path in a3m_by_stem.items():
                if protein_prefix in stem or stem in protein_prefix:
                    a3m_file = path
                    break
            # Fall back to order-based matching
            if a3m_file is None and i < len(a3m_files):
                a3m_file = a3m_files[i]
                print(f"WARNING: Using order-based matching for {seq_name} (index {i})")

        if a3m_file is None:
            print(f"WARNING: No a3m found for {seq_name}, skipping")
            skipped += 1
            continue

        # Create per-sequence directories
        seq_dir = seq_base / seq_name
        msa_dir = seq_dir / "msa"
        msa_dir.mkdir(parents=True, exist_ok=True)

        # Move a3m file (avoids storing MSA twice)
        dest_a3m = msa_dir / a3m_file.name
        shutil.move(str(a3m_file), dest_a3m)

        # Also move related files (.sto, .hhr, etc.)
        a3m_basename = a3m_file.stem
        for related in colabfold_path.glob(f"{a3m_basename}.*"):
            shutil.move(str(related), msa_dir / related.name)

        # Create YAML file for Boltz
        yaml_path = seq_dir / f"{seq_name}.yaml"
        # Use relative path so YAMLs stay valid when sequences are copied/subset
        a3m_relative = dest_a3m.relative_to(seq_dir)
        yaml_content = (
            f"version: 1\n"
            f"\n"
            f"sequences:\n"
            f"  - protein:\n"
            f'      id: "{protein_id}"\n'
            f"      sequence: {sequence}\n"
            f"      msa: {a3m_relative}\n"
        )
        yaml_path.write_text(yaml_content)

        print(f"Organized: {seq_name} -> {seq_dir}")
        processed += 1

    print(f"Processed: {processed}, Skipped: {skipped}")
    return processed, skipped


def main():
    parser = argparse.ArgumentParser(description="Scatter MSA outputs and create YAMLs")
    parser.add_argument("--file_list", required=True, help="Path to file_list.txt for this chunk")
    parser.add_argument("--colabfold_output", required=True, help="Path to colabfold output directory")
    parser.add_argument("--sequences_dir", required=True, help="Base sequences directory")
    args = parser.parse_args()

    processed, skipped = organize_chunk(
        args.file_list, args.colabfold_output, args.sequences_dir
    )
    if processed == 0:
        print("ERROR: No sequences were processed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
