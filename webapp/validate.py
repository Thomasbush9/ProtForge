"""
Input validation for FASTA and YAML files used by the ProtForge pipeline.
"""

import shutil
from pathlib import Path

import yaml

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")
# Only lowercase — the pipeline (chunk_fastas.py) only accepts lowercase extensions
FASTA_EXTENSIONS = {".fasta", ".fa"}
YAML_EXTENSIONS = {".yaml", ".yml"}


def validate_fasta(path: Path) -> dict:
    """Validate a single FASTA file.

    Pipeline expects exactly one sequence per FASTA file.
    Only lowercase extensions (.fasta, .fa) are accepted by the pipeline.

    Returns dict with: filename, valid, num_sequences, total_residues, errors.
    """
    result = {
        "filename": path.name,
        "path": str(path),
        "valid": False,
        "num_sequences": 0,
        "total_residues": 0,
        "errors": [],
    }

    # Reject uppercase extensions — pipeline won't pick them up
    if path.suffix not in FASTA_EXTENSIONS:
        result["errors"].append(
            f"Extension '{path.suffix}' not recognized by pipeline (must be lowercase .fasta or .fa)"
        )
        return result

    try:
        text = path.read_text()
    except Exception as e:
        result["errors"].append(f"Cannot read file: {e}")
        return result

    if not text.strip():
        result["errors"].append("File is empty")
        return result

    lines = text.strip().splitlines()
    if not lines[0].startswith(">"):
        result["errors"].append("First line must start with '>' (FASTA header)")
        return result

    seq_count = 0
    total_residues = 0
    current_seq = []
    current_header = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            # Flush previous sequence
            if current_seq:
                seq_str = "".join(current_seq)
                total_residues += len(seq_str)
                invalid_chars = set(seq_str.upper()) - VALID_AAS
                if invalid_chars:
                    result["errors"].append(
                        f"Sequence '{current_header}': invalid characters: {', '.join(sorted(invalid_chars))}"
                    )
            seq_count += 1
            current_header = line[1:].split()[0] if len(line) > 1 else f"seq_{seq_count}"
            current_seq = []
        else:
            current_seq.append(line.replace(" ", ""))

    # Flush last sequence
    if current_seq:
        seq_str = "".join(current_seq)
        total_residues += len(seq_str)
        invalid_chars = set(seq_str.upper()) - VALID_AAS
        if invalid_chars:
            result["errors"].append(
                f"Sequence '{current_header}': invalid characters: {', '.join(sorted(invalid_chars))}"
            )
    elif seq_count > 0:
        result["errors"].append(f"Sequence '{current_header}' has no residues")

    # Pipeline assumes one sequence per FASTA file
    if seq_count > 1:
        result["errors"].append(
            f"Contains {seq_count} sequences — pipeline expects exactly 1 per file"
        )

    result["num_sequences"] = seq_count
    result["total_residues"] = total_residues
    result["valid"] = seq_count == 1 and len(result["errors"]) == 0
    return result


def validate_yaml(path: Path) -> dict:
    """Validate a single Boltz YAML file.

    Expected schema:
        version: 1
        sequences:
          - protein:
              id: "..."
              sequence: "..."
              msa: "..." or "empty"

    Returns dict with: filename, valid, sequence_id, sequence_length, has_msa, errors.
    """
    result = {
        "filename": path.name,
        "path": str(path),
        "valid": False,
        "sequence_id": "",
        "sequence_length": 0,
        "has_msa": False,
        "errors": [],
    }

    # Only lowercase extensions
    if path.suffix not in YAML_EXTENSIONS:
        result["errors"].append(
            f"Extension '{path.suffix}' not recognized (must be lowercase .yaml or .yml)"
        )
        return result

    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        result["errors"].append(f"Invalid YAML: {e}")
        return result
    except Exception as e:
        result["errors"].append(f"Cannot read file: {e}")
        return result

    if not isinstance(data, dict):
        result["errors"].append("Top-level must be a mapping")
        return result

    if "version" not in data:
        result["errors"].append("Missing 'version' key")

    sequences = data.get("sequences")
    if not sequences:
        result["errors"].append("Missing or empty 'sequences' list")
        return result
    if not isinstance(sequences, list):
        result["errors"].append("'sequences' must be a list")
        return result

    first = sequences[0]
    if not isinstance(first, dict) or "protein" not in first:
        result["errors"].append("First sequence entry must have a 'protein' key")
        return result

    protein = first["protein"]
    if not isinstance(protein, dict):
        result["errors"].append("'protein' must be a mapping")
        return result

    for key in ("id", "sequence"):
        if key not in protein:
            result["errors"].append(f"Missing 'protein.{key}'")

    if "id" in protein:
        result["sequence_id"] = str(protein["id"])

    if "sequence" in protein:
        seq = str(protein["sequence"])
        result["sequence_length"] = len(seq)
        invalid_chars = set(seq.upper()) - VALID_AAS
        if invalid_chars:
            result["errors"].append(
                f"Invalid amino acids in sequence: {', '.join(sorted(invalid_chars))}"
            )

    msa_val = protein.get("msa", "")
    result["has_msa"] = bool(msa_val) and str(msa_val).lower() != "empty"

    result["valid"] = len(result["errors"]) == 0
    return result


def scan_directory(dir_path: Path) -> dict:
    """Scan a directory for FASTA and YAML files, validate each.

    Only picks up files with lowercase extensions (matching pipeline behavior).

    Returns dict with:
        file_type: "fasta" | "yaml" | "mixed" | "none"
        fasta_results: list of validate_fasta() results
        yaml_results: list of validate_yaml() results
        total_files: int
        valid_count: int
        invalid_count: int
    """
    fasta_files = []
    yaml_files = []

    for f in sorted(dir_path.iterdir()):
        if not f.is_file():
            continue
        ext = f.suffix  # No .lower() — match pipeline behavior exactly
        if ext in FASTA_EXTENSIONS:
            fasta_files.append(f)
        elif ext in YAML_EXTENSIONS:
            yaml_files.append(f)

    fasta_results = [validate_fasta(f) for f in fasta_files]
    yaml_results = [validate_yaml(f) for f in yaml_files]

    all_results = fasta_results + yaml_results
    valid_count = sum(1 for r in all_results if r["valid"])
    invalid_count = sum(1 for r in all_results if not r["valid"])

    if fasta_results and not yaml_results:
        file_type = "fasta"
    elif yaml_results and not fasta_results:
        file_type = "yaml"
    elif fasta_results and yaml_results:
        file_type = "mixed"
    else:
        file_type = "none"

    return {
        "file_type": file_type,
        "fasta_results": fasta_results,
        "yaml_results": yaml_results,
        "total_files": len(all_results),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
    }


def copy_valid_files(scan_result: dict, dest_dir: Path) -> int:
    """Copy only valid files from a scan result into dest_dir.

    Returns the number of files copied.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for results in (scan_result["fasta_results"], scan_result["yaml_results"]):
        for r in results:
            if r["valid"]:
                src = Path(r["path"])
                shutil.copy2(src, dest_dir / src.name)
                copied += 1
    return copied
