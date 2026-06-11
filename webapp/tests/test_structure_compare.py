"""
Tests for the structure-comparison core (webapp/structure_compare.py).

The mmCIF CA parser and the Kabsch RMSD are pure NumPy and fully tested here.
TM-score goes through tmtools (optional) and isn't exercised in CI.

Run with:
    PYTHONPATH=webapp python -m pytest webapp/tests/test_structure_compare.py -v
"""

import numpy as np
import pytest

from structure_compare import ca_coords_and_seq, kabsch_rmsd, compare


def _mini_cif(residues_xyz):
    """Build a tiny mmCIF with CA atoms (and a non-CA + a HETATM to be skipped).

    residues_xyz: list of (3-letter resname, (x, y, z)).
    """
    header = (
        "data_t\nloop_\n"
        "_atom_site.group_PDB\n_atom_site.label_atom_id\n_atom_site.label_comp_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "_atom_site.pdbx_PDB_model_num\n"
    )
    rows = []
    for i, (res, (x, y, z)) in enumerate(residues_xyz, 1):
        # A non-CA atom for the same residue (must be ignored) + the CA.
        rows.append(f"ATOM N {res} {x} {y} {z} 1")
        rows.append(f"ATOM CA {res} {x} {y} {z} 1")
    rows.append("HETATM CA HOH 0 0 0 1")   # waters etc. must be skipped
    return header + "\n".join(rows) + "\n#\n"


def test_ca_parser_extracts_ca_and_sequence():
    cif = _mini_cif([("MET", (0.0, 0.0, 0.0)),
                     ("ALA", (1.0, 0.0, 0.0)),
                     ("GLY", (2.0, 0.0, 0.0))])
    coords, seq = ca_coords_and_seq(cif)
    assert seq == "MAG"                       # only CA, standard residues
    assert coords.shape == (3, 3)
    assert np.allclose(coords[1], [1.0, 0.0, 0.0])


def test_ca_parser_stops_at_second_model():
    cif = (
        "data_t\nloop_\n_atom_site.group_PDB\n_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n_atom_site.pdbx_PDB_model_num\n"
        "ATOM CA MET 0 0 0 1\n"
        "ATOM CA ALA 1 0 0 2\n"   # model 2 — must be ignored
        "#\n"
    )
    _, seq = ca_coords_and_seq(cif)
    assert seq == "M"


def test_kabsch_rmsd_zero_for_rigid_transform():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(12, 3))
    # Rotate + translate P -> Q; optimal superposition RMSD must be ~0.
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    Q = P @ R.T + np.array([5.0, -2.0, 1.0])
    assert kabsch_rmsd(P, Q) == pytest.approx(0.0, abs=1e-9)


def test_kabsch_rmsd_nonzero_and_length_guard():
    P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    Q = P.copy()
    Q[0] += np.array([3.0, 0, 0])  # perturb one point — no rigid fit removes it
    assert kabsch_rmsd(P, Q) > 0.0
    with pytest.raises(ValueError):
        kabsch_rmsd(P, P[:3])


def test_compare_equal_length_gives_rmsd():
    a = _mini_cif([("MET", (0.0, 0.0, 0.0)), ("ALA", (1.0, 0.0, 0.0))])
    b = _mini_cif([("MET", (0.0, 0.0, 0.0)), ("ALA", (1.0, 0.0, 0.0))])
    out = compare(a, b)
    assert out["target_len"] == out["query_len"] == 2
    assert out["rmsd"] == pytest.approx(0.0, abs=1e-9)


def test_compare_unequal_length_notes_rmsd_unavailable():
    a = _mini_cif([("MET", (0.0, 0.0, 0.0)), ("ALA", (1.0, 0.0, 0.0))])
    b = _mini_cif([("MET", (0.0, 0.0, 0.0))])
    out = compare(a, b)
    assert out["rmsd"] is None
    assert any("residue counts differ" in n for n in out["notes"])
