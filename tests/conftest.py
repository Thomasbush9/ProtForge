"""Shared fixtures for workflow script tests."""

from pathlib import Path


def make_fasta(directory: Path, name: str, sequence: str, header: str | None = None) -> Path:
    """Create a .fasta file with given name and sequence."""
    fasta_path = directory / f"{name}.fasta"
    hdr = header if header else f">{name}"
    fasta_path.write_text(f"{hdr}\n{sequence}\n")
    return fasta_path


def make_yaml(directory: Path, name: str, sequence: str, msa_path: str = "empty") -> Path:
    """Create a Boltz-format .yaml file."""
    yaml_path = directory / f"{name}.yaml"
    yaml_path.write_text(
        f"version: 1\n\n"
        f"sequences:\n"
        f"  - protein:\n"
        f'      id: "{name}"\n'
        f"      sequence: {sequence}\n"
        f"      msa: {msa_path}\n"
    )
    return yaml_path


def make_a3m(directory: Path, name: str, homologs: bool = True) -> Path:
    """Create a dummy .a3m file. If homologs=False, only the query sequence (no real MSA)."""
    a3m_path = directory / f"{name}.a3m"
    if homologs:
        a3m_path.write_text(f">{name}\nMKTLAEEL\n>homolog1\nMRTLAEEL\n")
    else:
        a3m_path.write_text(f">{name}\nMKTLAEEL\n>{name}\nMKTLAEEL\n")
    return a3m_path


def make_boltz_prediction(directory: Path, seq_id: str, model_numbers: list[int]) -> Path:
    """Create a fake Boltz prediction directory with _model_N.cif files."""
    pred_dir = directory / seq_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    for n in model_numbers:
        (pred_dir / f"{seq_id}_model_{n}.cif").write_text(f"MODEL {n}")
        (pred_dir / f"{seq_id}_confidence_model_{n}.json").write_text(f'{{"model": {n}}}')
    return pred_dir
