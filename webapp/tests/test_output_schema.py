"""
Tests for the normalized output schema shared by the structure predictors
(workflow/scripts/output_schema.py).

Cover: per-stage model ids, confidence extraction onto the uniform schema,
pLDDT normalization (0-1 -> 0-100), B-factor pLDDT detection, and the sidecar
write / prefer-sidecar round-trip.

Run with:
    PYTHONPATH=webapp:workflow/scripts python -m pytest webapp/tests/test_output_schema.py -v
"""

import json

import pytest

import output_schema as osch


# --- helpers --------------------------------------------------------------


def _cif_with_bfactors(values):
    header = (
        "data_test\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.B_iso_or_equiv\n"
    )
    rows = "".join(f"ATOM {i} C {v}\n" for i, v in enumerate(values, 1))
    return header + rows


def _pdb_line(bfac):
    line = list(" " * 80)
    line[0:6] = list("ATOM  ")
    line[60:66] = list(f"{bfac:6.2f}")
    return "".join(line)


# --- model ids ------------------------------------------------------------


def test_model_id_for(tmp_path):
    assert osch.model_id_for("boltz", tmp_path / "mut_model_0.cif") == "model_0"
    assert osch.model_id_for(
        "openfold", tmp_path / "mut_seed_42_sample_3_model.cif") == "seed_42_sample_3"
    fast = tmp_path / "esmfold" / "fast"
    fast.mkdir(parents=True)
    assert osch.model_id_for("esmfold", fast / "structure.cif") == "fast"


# --- B-factor detection ---------------------------------------------------


def test_bfactor_detection_cif(tmp_path):
    varied = tmp_path / "v.cif"
    varied.write_text(_cif_with_bfactors([80.5, 65.2, 90.1, 55.0]))
    assert osch.structure_plddt_in_bfactor(varied) is True

    flat = tmp_path / "f.cif"
    flat.write_text(_cif_with_bfactors([0.0, 0.0, 0.0]))
    assert osch.structure_plddt_in_bfactor(flat) is False


def test_bfactor_detection_pdb(tmp_path):
    p = tmp_path / "m.pdb"
    p.write_text("\n".join(_pdb_line(b) for b in (82.5, 70.1, 91.0)) + "\n")
    assert osch.structure_plddt_in_bfactor(p) is True


# --- extraction onto the uniform schema -----------------------------------


def test_extract_boltz(tmp_path):
    d = tmp_path / "boltz"
    d.mkdir()
    (d / "mut_model_0.cif").write_text(_cif_with_bfactors([80.0, 90.0, 70.0]))
    # Boltz reports complex_plddt on a 0-1 scale -> should normalize to 0-100.
    (d / "confidence_mut_model_0.json").write_text(json.dumps(
        {"confidence_score": 0.88, "ptm": 0.81, "iptm": 0.4, "complex_plddt": 0.82}))

    s = osch.extract_summary("boltz", d / "mut_model_0.cif")
    assert s.stage == "boltz" and s.model_id == "model_0"
    assert s.ranking_score == 0.88
    assert s.ptm == 0.81 and s.iptm == 0.4
    assert s.plddt_mean == pytest.approx(82.0)  # 0.82 -> 82
    assert s.plddt_in_bfactor is True


def test_boltz_multi_model_picks_its_own_confidence(tmp_path):
    # Two models in one dir, real Boltz naming. Each must get ITS OWN confidence
    # JSON — the sequence prefix ("34073") alone matches both, so matching on it
    # would cross-wire model_1 to model_0's scores. Regression for that bug.
    d = tmp_path / "boltz"
    d.mkdir()
    for n, ptm in ((0, 0.93), (1, 0.41)):
        (d / f"34073_model_{n}.cif").write_text(_cif_with_bfactors([80.0, 90.0]))
        (d / f"confidence_34073_model_{n}.json").write_text(
            json.dumps({"ptm": ptm, "confidence_score": 0.5 + n, "complex_plddt": 0.9}))

    s0 = osch.extract_summary("boltz", d / "34073_model_0.cif")
    s1 = osch.extract_summary("boltz", d / "34073_model_1.cif")
    assert (s0.model_id, s0.ptm) == ("model_0", 0.93)
    assert (s1.model_id, s1.ptm) == ("model_1", 0.41)


def test_extract_openfold(tmp_path):
    d = tmp_path / "openfold"
    d.mkdir()
    (d / "mut_seed_1_sample_0_model.cif").write_text(_cif_with_bfactors([60.0, 75.0]))
    (d / "mut_seed_1_sample_0_confidences_aggregated.json").write_text(json.dumps(
        {"ranking_score": 0.74, "ptm": 0.7, "avg_plddt": 88.0}))

    s = osch.extract_summary("openfold", d / "mut_seed_1_sample_0_model.cif")
    assert s.model_id == "seed_1_sample_0"
    assert s.ranking_score == 0.74
    assert s.plddt_mean == pytest.approx(88.0)  # already 0-100, unchanged


def test_extract_esmfold(tmp_path):
    np = pytest.importorskip("numpy")
    fast = tmp_path / "esmfold" / "fast"
    fast.mkdir(parents=True)
    fast.joinpath("structure.cif").write_text(_cif_with_bfactors([0.7, 0.9]))
    np.save(fast / "plddt.npy", np.array([0.80, 0.90, 0.70]))  # 0-1 scale

    s = osch.extract_summary("esmfold", fast / "structure.cif")
    assert s.model_id == "fast"
    assert s.plddt_mean == pytest.approx(80.0)  # mean 0.8 -> 80


# --- sidecar round-trip ---------------------------------------------------


def test_sidecar_write_and_prefer(tmp_path):
    d = tmp_path / "boltz"
    d.mkdir()
    struct = d / "mut_model_0.cif"
    struct.write_text(_cif_with_bfactors([80.0, 90.0]))
    (d / "confidence_mut_model_0.json").write_text(json.dumps({"ptm": 0.5}))

    path = osch.write_summary_for("boltz", struct)
    assert path.name == "model_0.summary.json"
    assert path.is_file()

    # summary_for_structure should now return the sidecar's content verbatim.
    on_disk = json.loads(path.read_text())
    on_disk["ptm"] = 0.999  # tamper so we can tell sidecar vs live extraction
    path.write_text(json.dumps(on_disk))
    got = osch.summary_for_structure("boltz", struct)
    assert got["ptm"] == 0.999
