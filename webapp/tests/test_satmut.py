"""
Tests for the saturation-mutagenesis core (webapp/satmut.py).

The launch path is cluster-side; here we cover the testable parts: the FASTA
write, deriving the LLR matrix from a synthetic logits.npy, the CSV round-trip
via load_matrix, and the shape of the sbatch command.

Run with:
    PYTHONPATH=webapp:workflow/scripts python -m pytest webapp/tests/test_satmut.py -v
"""

import json

import numpy as np
import pytest

import satmut


def test_write_query_fasta(tmp_path):
    p = satmut.write_query_fasta(tmp_path, "gfp", "MSKG")
    assert p == tmp_path / "input" / "gfp.fasta"
    assert p.read_text() == ">gfp\nMSKG\n"


_CANON = "ACDEFGHIKLMNPQRSTVWY"


def _fake_logits(out_dir, name, size, aa_token_ids, logits):
    sd = satmut.size_dir(out_dir, name, size)
    sd.mkdir(parents=True)
    np.save(sd / "logits.npy", np.asarray(logits, dtype=np.float16))
    (sd / "aa_token_ids.json").write_text(json.dumps(aa_token_ids))


def test_derive_and_load_roundtrip(tmp_path):
    # Realistic vocab: all 20 canonical AAs at columns 0..19 (as the runner saves).
    aa = {a: i for i, a in enumerate(_CANON)}
    logits = np.zeros((2, 20), dtype=float)
    logits[0, aa["C"]] = 3.0   # pos0 wt=A: LLR(A->C) = 3 - 0 = +3
    logits[1, aa["D"]] = -4.0  # pos1 wt=C: LLR(C->D) = -4 - 0 = -4
    _fake_logits(tmp_path, "q", "6B", aa, logits)

    csv_path = satmut.derive_scan(tmp_path, "q", "6B", "AC")
    assert csv_path == satmut.matrix_csv(tmp_path, "q", "6B")
    assert csv_path.is_file()

    wt, aa_order, mat = satmut.load_matrix(csv_path)
    assert wt == "AC"
    assert aa_order == list(_CANON)
    assert mat[0, aa_order.index("A")] == pytest.approx(0.0)   # wt col is 0
    assert mat[0, aa_order.index("C")] == pytest.approx(3.0)
    assert mat[1, aa_order.index("D")] == pytest.approx(-4.0)


def test_derive_scan_missing_logits_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        satmut.derive_scan(tmp_path, "q", "6B", "AC")


def test_build_sbatch_command_uses_config(tmp_path):
    cfg = {
        "containers": {"esmc": "/img/esm.sif", "runtime": "apptainer"},
        "esmc": {"cache_dir": "/hf/cache"},
        "slurm": {"partition": "kempner_h100", "account": "lab", "log_dir": "/logs"},
    }
    cmd = satmut.build_sbatch_command(cfg, tmp_path, "gfp", "6B")
    assert cmd[0] == "sbatch"
    # Key flags / args present.
    assert "--partition" in cmd and "kempner_h100" in cmd
    assert "--account" in cmd and "lab" in cmd
    assert "--gpus" in cmd
    assert any(c.endswith("run_satmut.sh") for c in cmd)
    assert "--size" in cmd and "6B" in cmd
    assert "/img/esm.sif" in cmd and "/hf/cache" in cmd
    assert "apptainer" in cmd  # runtime passed through


def test_resolve_cluster_settings_reports_missing():
    s = satmut.resolve_cluster_settings({})
    assert s["errors"]  # sif, cache, partition all missing
    assert any("containers.esmc" in e for e in s["errors"])
