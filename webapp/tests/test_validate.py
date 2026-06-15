"""
Tests for input validation (webapp/validate.py) and the launch-input check
(webapp/pipeline_ops.py).

Covers: U (selenocysteine) is accepted across FASTA/YAML validation, and the
spot-and-exclude flow that lets the pipeline skip invalid files instead of
blocking the whole run.

Run with:
    PYTHONPATH=webapp:workflow/scripts python -m pytest webapp/tests/test_validate.py -v
"""

import yaml

from validate import (
    VALID_AAS,
    INVALID_SUFFIX,
    validate_fasta,
    validate_yaml,
    scan_directory,
    exclude_invalid_files,
)
from pipeline_ops import check_launch_inputs, validate_launch_inputs


def _write_fasta(path, header, seq):
    path.write_text(f">{header}\n{seq}\n")


def _write_yaml(path, seq, seq_id="x"):
    path.write_text(yaml.safe_dump({
        "version": 1,
        "sequences": [{"protein": {"id": seq_id, "sequence": seq, "msa": "empty"}}],
    }))


# --------------------------------------------------------------------------
# U (selenocysteine) is allowed
# --------------------------------------------------------------------------

def test_U_is_in_valid_alphabet():
    assert "U" in VALID_AAS


def test_fasta_with_U_is_valid(tmp_path):
    p = tmp_path / "sec.fasta"
    _write_fasta(p, "sec", "MAUKGU")
    result = validate_fasta(p)
    assert result["valid"], result["errors"]
    assert result["total_residues"] == 6


def test_yaml_with_U_is_valid(tmp_path):
    p = tmp_path / "sec.yaml"
    _write_yaml(p, "MAUKGU")
    result = validate_yaml(p)
    assert result["valid"], result["errors"]


def test_fasta_with_truly_invalid_char_still_fails(tmp_path):
    p = tmp_path / "bad.fasta"
    _write_fasta(p, "bad", "MAXKG")  # X is not allowed
    result = validate_fasta(p)
    assert not result["valid"]
    assert any("X" in e for e in result["errors"])


# --------------------------------------------------------------------------
# exclude_invalid_files
# --------------------------------------------------------------------------

def test_exclude_renames_only_invalid(tmp_path):
    good = tmp_path / "good.fasta"
    bad = tmp_path / "bad.fasta"
    _write_fasta(good, "good", "MAUKG")
    _write_fasta(bad, "bad", "MAXKG")

    scan = scan_directory(tmp_path)
    renamed = exclude_invalid_files(scan)

    assert renamed == [str(bad) + INVALID_SUFFIX]
    assert good.exists()
    assert not bad.exists()
    assert (tmp_path / ("bad.fasta" + INVALID_SUFFIX)).exists()


def test_excluded_file_no_longer_scanned(tmp_path):
    _write_fasta(tmp_path / "good.fasta", "good", "MAUKG")
    _write_fasta(tmp_path / "bad.fasta", "bad", "MAXKG")

    exclude_invalid_files(scan_directory(tmp_path))

    rescan = scan_directory(tmp_path)
    assert rescan["total_files"] == 1
    assert rescan["invalid_count"] == 0
    assert rescan["valid_count"] == 1


# --------------------------------------------------------------------------
# check_launch_inputs — spot invalid without blocking
# --------------------------------------------------------------------------

def test_msa_invalid_files_are_skippable_not_fatal(tmp_path):
    _write_fasta(tmp_path / "good.fasta", "good", "MAUKG")
    _write_fasta(tmp_path / "bad.fasta", "bad", "MAXKG")
    cfg = {"pipeline": {"msa": True}, "input": {"fasta_dir": str(tmp_path)}}

    check = check_launch_inputs(cfg)
    assert check["errors"] == []  # not fatal — there is a valid file
    assert len(check["invalid_groups"]) == 1
    assert len(check["invalid_groups"][0]["invalid"]) == 1


def test_msa_all_invalid_is_fatal(tmp_path):
    _write_fasta(tmp_path / "bad.fasta", "bad", "MAXKG")
    cfg = {"pipeline": {"msa": True}, "input": {"fasta_dir": str(tmp_path)}}

    check = check_launch_inputs(cfg)
    assert check["errors"], "all-invalid should block launch"


def test_msa_missing_dir_is_fatal(tmp_path):
    cfg = {"pipeline": {"msa": True}, "input": {"fasta_dir": str(tmp_path / "nope")}}
    assert check_launch_inputs(cfg)["errors"]


def test_msa_off_yaml_invalid_files_are_skippable(tmp_path):
    _write_yaml(tmp_path / "good.yaml", "MAUKG")
    (tmp_path / "bad.yaml").write_text("not: [valid")  # broken YAML
    cfg = {
        "pipeline": {"msa": False, "boltz": True},
        "input": {"yaml_dir": str(tmp_path)},
    }
    check = check_launch_inputs(cfg)
    assert check["errors"] == []
    assert len(check["invalid_groups"]) == 1


def test_validate_launch_inputs_wrapper_returns_errors_only(tmp_path):
    _write_fasta(tmp_path / "good.fasta", "good", "MAUKG")
    _write_fasta(tmp_path / "bad.fasta", "bad", "MAXKG")
    cfg = {"pipeline": {"msa": True}, "input": {"fasta_dir": str(tmp_path)}}
    # Back-compat: invalid-but-recoverable inputs no longer appear as errors.
    assert validate_launch_inputs(cfg) == []
