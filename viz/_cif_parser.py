"""Lightweight mmCIF parser for extracting CA coordinates and pLDDT.

No external dependencies beyond numpy and pandas. Parses the ``_atom_site``
loop from mmCIF files produced by Boltz (or any standard mmCIF).

Provides:
- ``ProteinStructure`` — lightweight dataclass holding coords/sequence/pLDDT
- ``parse_cif()`` — parse a CIF file into a ProteinStructure
- ``load_plddt_df()`` — extract pLDDT as a DataFrame (same shape as ES CSVs)
- ``load_confidence_dfs()`` — batch-load pLDDT from a sequences directory
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Standard amino acid 3-letter → 1-letter mapping
_AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def _parse_3letter(code: str) -> str:
    """Convert 3-letter amino acid code to single letter."""
    return _AA_3TO1.get(code.upper(), "X")


# ---------------------------------------------------------------------------
# ProteinStructure dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProteinStructure:
    """Lightweight container for protein structure data.

    Attributes
    ----------
    coord : np.ndarray, shape (N, 3)
        Alpha-carbon coordinates.
    sequence : np.ndarray, shape (N,)
        Single-letter amino acid codes.
    plddt : np.ndarray, shape (N,)
        Per-residue pLDDT confidence scores (0–100).
    idx : np.ndarray, shape (N,)
        Residue indices (1-based).
    """

    coord: np.ndarray
    sequence: np.ndarray
    plddt: np.ndarray
    idx: np.ndarray

    @property
    def seq_len(self) -> int:
        return len(self.idx)


# ---------------------------------------------------------------------------
# mmCIF _atom_site loop parser (pure Python, no Biopython)
# ---------------------------------------------------------------------------

def _parse_atom_site_loop(path: str | Path) -> pd.DataFrame:
    """Parse the ``_atom_site`` loop from an mmCIF file into a DataFrame.

    Handles the standard loop_ format where field names are listed first
    (``_atom_site.field_name``), followed by whitespace-delimited data rows.
    """
    path = Path(path)
    lines = path.read_text().splitlines()

    # Find the loop_ block that contains _atom_site fields
    fields = []
    data_start = None
    in_atom_site = False

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect start of a loop containing _atom_site
        if line == "loop_":
            # Peek ahead for _atom_site fields
            j = i + 1
            candidate_fields = []
            while j < len(lines) and lines[j].strip().startswith("_atom_site."):
                field_name = lines[j].strip().split(".")[1]
                candidate_fields.append(field_name)
                j += 1
            if candidate_fields:
                fields = candidate_fields
                data_start = j
                in_atom_site = True
                break
        i += 1

    if not fields or data_start is None:
        raise ValueError(f"No _atom_site loop found in {path}")

    # Read data rows until we hit a non-data line
    rows = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("#") or line.startswith("_") or line == "loop_":
            break
        # Split by whitespace (handles standard mmCIF atom_site rows)
        tokens = line.split()
        if len(tokens) == len(fields):
            rows.append(tokens)
        elif len(tokens) > len(fields):
            # Some CIF files have extra whitespace; take first N tokens
            rows.append(tokens[: len(fields)])

    return pd.DataFrame(rows, columns=fields)


def parse_cif(path: str | Path, chain: str = "", min_plddt: float = 0.0) -> ProteinStructure:
    """Parse an mmCIF file into a ``ProteinStructure``.

    Parameters
    ----------
    path : str or Path
        Path to the CIF file.
    chain : str, optional
        Specific chain to extract (empty string = all chains).
    min_plddt : float, optional
        Residues with pLDDT <= this value get NaN coordinates.

    Returns
    -------
    ProteinStructure
    """
    df = _parse_atom_site_loop(path)

    # Filter to CA atoms, model 1, valid residue IDs
    mask = df["label_atom_id"] == "CA"
    if "pdbx_PDB_model_num" in df.columns:
        mask &= df["pdbx_PDB_model_num"] == "1"
    if "label_seq_id" in df.columns:
        mask &= df["label_seq_id"] != "."
    if "label_alt_id" in df.columns:
        mask &= df["label_alt_id"] != "B"
    df = df[mask].copy()

    if chain and "label_asym_id" in df.columns:
        df = df[df["label_asym_id"] == chain]

    if df.empty:
        raise ValueError(f"No CA atoms found in {path}")

    coord = np.column_stack([
        df["Cartn_x"].values.astype(float),
        df["Cartn_y"].values.astype(float),
        df["Cartn_z"].values.astype(float),
    ])
    idx = df["label_seq_id"].values.astype(int)
    sequence = np.array([_parse_3letter(aa) for aa in df["label_comp_id"]])
    plddt = df["B_iso_or_equiv"].values.astype(float)

    # Apply pLDDT filter: set low-confidence coords to NaN
    if min_plddt > 0:
        low_conf = plddt <= min_plddt
        coord[low_conf] = np.nan

    return ProteinStructure(coord=coord, sequence=sequence, plddt=plddt, idx=idx)


# ---------------------------------------------------------------------------
# DataFrame loaders (for dashboard integration)
# ---------------------------------------------------------------------------

def load_plddt_df(cif_path: str | Path) -> pd.DataFrame:
    """Extract pLDDT from a CIF file as a DataFrame.

    Returns a DataFrame with columns ``residue_index`` and ``plddt``,
    matching the format of ES output CSVs for dashboard compatibility.
    """
    prot = parse_cif(cif_path)
    return pd.DataFrame({
        "residue_index": prot.idx,
        "plddt": prot.plddt,
        "protA_resname": prot.sequence,
    })


def _find_best_cif(boltz_dir: Path) -> Path | None:
    """Find the best CIF file in a boltz output directory.

    Tries paths.txt first (written by collect_cif_paths.py), then
    falls back to scanning run_0/ or the directory directly.
    """
    # Try paths.txt first
    paths_file = boltz_dir / "paths.txt"
    if paths_file.exists():
        lines = [l.strip() for l in paths_file.read_text().splitlines() if l.strip()]
        if lines:
            p = Path(lines[0])
            if p.exists():
                return p

    # Fallback: scan run_0/ or directory directly
    run0 = boltz_dir / "run_0"
    search_dir = run0 if run0.is_dir() else boltz_dir
    cifs = sorted(search_dir.glob("*_model_*.cif"))
    if cifs:
        return cifs[-1]
    cifs = sorted(search_dir.glob("*.cif"))
    return cifs[-1] if cifs else None


def load_confidence_dfs(sequences_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load pLDDT data from CIF files for all sequences.

    Parameters
    ----------
    sequences_dir : str or Path
        ProtForge sequences directory (contains ``{name}/boltz/`` subdirs).

    Returns
    -------
    dict[str, DataFrame]
        Mapping of sequence name to DataFrame with ``residue_index``
        and ``plddt`` columns.
    """
    sequences_dir = Path(sequences_dir)
    result = {}
    for seq_dir in sorted(sequences_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        boltz_dir = seq_dir / "boltz"
        if not boltz_dir.is_dir():
            continue
        cif = _find_best_cif(boltz_dir)
        if cif is None:
            continue
        try:
            result[seq_dir.name] = load_plddt_df(cif)
        except (ValueError, KeyError):
            continue
    return result
