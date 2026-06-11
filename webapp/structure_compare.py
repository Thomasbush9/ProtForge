"""
Structure-to-structure comparison for the Results tab.

Compute similarity between predicted structures — one target vs one or more
queries — as:
  - RMSD: optimal-superposition CA RMSD via Kabsch (pure NumPy). Assumes a
    residue-by-residue (1:1) correspondence, so it's valid when the structures
    are predictions of the same / point-mutated sequence (equal length). No
    extra dependency.
  - TM-score: sequence-independent structural alignment via `tmtools` (US-align
    TM-align). Handles different lengths / indels. Optional — returns None when
    tmtools isn't installed; the view degrades to RMSD only.

CA coordinates + sequence are parsed straight from the mmCIF _atom_site loop
(header-indexed, so column order doesn't matter), no Bio/biotite dependency.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def read_cif_text(path: str | Path) -> str:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return f.read()
    return path.read_text(errors="replace")


def _atom_site_loop(lines: list[str]) -> tuple[dict[str, int], int] | None:
    """Locate the _atom_site loop; return ({column_name: index}, data_start_line)."""
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() == "loop_":
            headers, j = [], i + 1
            while j < n and lines[j].lstrip().startswith("_"):
                headers.append(lines[j].strip())
                j += 1
            if any(h.startswith("_atom_site.") for h in headers):
                cols = {h: k for k, h in enumerate(headers)}
                return cols, j
            i = j
        else:
            i += 1
    return None


def ca_coords_and_seq(cif_text: str) -> tuple[np.ndarray, str]:
    """Return (CA coordinates (N, 3), one-letter sequence) for the first model.

    Only standard-residue ATOM CA records are used; ligands/hetero/unknown
    residues are skipped. Coordinates are in file (residue) order.
    """
    lines = cif_text.splitlines()
    loop = _atom_site_loop(lines)
    if loop is None:
        return np.empty((0, 3)), ""
    cols, start = loop
    try:
        c_group = cols["_atom_site.group_PDB"]
        c_atom = cols["_atom_site.label_atom_id"]
        c_comp = cols["_atom_site.label_comp_id"]
        c_x = cols["_atom_site.Cartn_x"]
        c_y = cols["_atom_site.Cartn_y"]
        c_z = cols["_atom_site.Cartn_z"]
    except KeyError:
        return np.empty((0, 3)), ""
    c_model = cols.get("_atom_site.pdbx_PDB_model_num")

    coords, seq = [], []
    first_model = None
    for k in range(start, len(lines)):
        row = lines[k].strip()
        if not row or row.startswith(("_", "#", "loop_")):
            break
        f = row.split()
        if len(f) <= max(c_group, c_atom, c_comp, c_x, c_y, c_z):
            continue
        if f[c_group] != "ATOM" or f[c_atom].strip('"') != "CA":
            continue
        if c_model is not None and len(f) > c_model:
            if first_model is None:
                first_model = f[c_model]
            elif f[c_model] != first_model:
                break  # second model — stop at the first
        one = _THREE_TO_ONE.get(f[c_comp].upper())
        if one is None:
            continue
        try:
            coords.append((float(f[c_x]), float(f[c_y]), float(f[c_z])))
        except ValueError:
            continue
        seq.append(one)
    return np.asarray(coords, dtype=float), "".join(seq)


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Minimal RMSD between two equal-length point sets after optimal rigid
    superposition (Kabsch). P[i] is assumed to correspond to Q[i]."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[0] == 0:
        raise ValueError(f"need equal-length (N,3) arrays; got {P.shape} vs {Q.shape}")
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    Pr = Pc @ R.T
    return float(np.sqrt(np.mean(np.sum((Pr - Qc) ** 2, axis=1))))


def tm_score(coords_t: np.ndarray, seq_t: str,
             coords_q: np.ndarray, seq_q: str) -> dict | None:
    """TM-score + TM-align RMSD via tmtools. None if tmtools isn't installed.

    Returns tm normalized by the target length and by the query length (TM-score
    is asymmetric), plus the alignment RMSD.
    """
    try:
        import tmtools
    except ImportError:
        return None
    res = tmtools.tm_align(np.asarray(coords_t, float), np.asarray(coords_q, float),
                           seq_t, seq_q)
    return {
        "tm_norm_target": float(res.tm_norm_chain1),
        "tm_norm_query": float(res.tm_norm_chain2),
        "tm_rmsd": float(res.rmsd),
    }


def compare(target_cif_text: str, query_cif_text: str) -> dict:
    """Compare two structures. Always returns lengths; RMSD when residue counts
    match (1:1 CA correspondence); TM-score when tmtools is available."""
    ct, st = ca_coords_and_seq(target_cif_text)
    cq, sq = ca_coords_and_seq(query_cif_text)
    out: dict = {
        "target_len": len(st), "query_len": len(sq),
        "rmsd": None, "tm_norm_target": None, "tm_norm_query": None,
        "tm_rmsd": None, "notes": [],
    }
    if len(st) == 0 or len(sq) == 0:
        out["notes"].append("could not parse CA atoms from one of the structures")
        return out

    if len(st) == len(sq):
        out["rmsd"] = kabsch_rmsd(ct, cq)
        if st != sq:
            out["notes"].append(
                "RMSD assumes residue-by-residue correspondence; sequences "
                "differ, so it's only meaningful for point mutants of the same length."
            )
    else:
        out["notes"].append(
            f"residue counts differ ({len(st)} vs {len(sq)}); RMSD needs equal "
            "length — use TM-score (alignment-based)."
        )

    tm = tm_score(ct, st, cq, sq)
    if tm is None:
        out["notes"].append("tmtools not installed — TM-score unavailable "
                            "(`pip install tmtools`).")
    else:
        out.update(tm)
    return out
