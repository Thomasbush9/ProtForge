"""
Tests for the Results data layer (webapp/results.py).

Cover the two pure readers the Results tab depends on: structure discovery
across stages and Snakemake benchmark aggregation.

Run with:
    PYTHONPATH=webapp python -m pytest webapp/tests/test_results.py -v
"""

import json

from results import (
    list_result_sequences,
    list_sequence_dirs,
    find_structures,
    read_confidence,
    read_structure_text,
    structure_format,
    read_benchmarks,
)


def _make_output_tree(root):
    """Build a minimal output tree with one sequence and benchmarks."""
    seq = root / "sequences" / "mutant_A"
    (seq / "boltz").mkdir(parents=True)
    (seq / "openfold").mkdir(parents=True)
    (seq / "esmfold" / "fast").mkdir(parents=True)

    # Boltz: one model cif + sibling confidence json
    (seq / "boltz" / "mutant_A_model_0.cif").write_text("data_x\n_atom_site.\n")
    (seq / "boltz" / "confidence_mutant_A_model_0.json").write_text(
        json.dumps({"confidence_score": 0.87, "ptm": 0.81, "model": "boltz2"}))

    # OpenFold: one model cif + aggregated confidence json
    (seq / "openfold" / "mutant_A_seed_42_sample_0_model.cif").write_text("data_y\n")
    (seq / "openfold" / "mutant_A_seed_42_sample_0_confidences_aggregated.json").write_text(
        json.dumps({"ranking_score": 0.74}))

    # ESMFold: structure.cif (plddt.npy optional, skipped to avoid numpy dep)
    (seq / "esmfold" / "fast" / "structure.cif").write_text("data_z\n")

    # Benchmarks: two boltz TSVs, one msa TSV
    bench = root / "benchmarks"
    (bench / "boltz").mkdir(parents=True)
    (bench / "msa").mkdir(parents=True)
    _write_tsv(bench / "boltz" / "predict_0_run_0.tsv", s=120.0, max_rss=8000)
    _write_tsv(bench / "boltz" / "predict_1_run_0.tsv", s=180.0, max_rss=10000)
    _write_tsv(bench / "msa" / "colabfold_search_0.tsv", s=600.0, max_rss=240000)
    return root


def _write_tsv(path, s, max_rss):
    header = "s\th:m:s\tmax_rss\tmax_vms\tmax_uss\tmax_pss\tio_in\tio_out\tmean_load\tcpu_time"
    row = f"{s}\t0:02:00\t{max_rss}\t0\t0\t0\t0\t0\t0\t0"
    path.write_text(header + "\n" + row + "\n")


def test_list_and_find_structures(tmp_path):
    _make_output_tree(tmp_path)

    seqs = list_result_sequences(tmp_path)
    assert seqs == ["mutant_A"]

    # Cheap listing returns every dir (incl. result-less ones) without globbing.
    (tmp_path / "sequences" / "running_B").mkdir()
    assert list_sequence_dirs(tmp_path) == ["mutant_A", "running_B"]
    # list_result_sequences still filters to those with structures.
    assert list_result_sequences(tmp_path) == ["mutant_A"]

    structures = find_structures(tmp_path / "sequences" / "mutant_A")
    stages = {s.stage for s in structures}
    assert stages == {"boltz", "openfold", "esmfold"}
    assert len(structures) == 3


def test_read_confidence_boltz_and_openfold(tmp_path):
    _make_output_tree(tmp_path)
    structures = {s.stage: s for s in find_structures(tmp_path / "sequences" / "mutant_A")}

    boltz_conf = read_confidence(structures["boltz"])
    assert boltz_conf["confidence_score"] == 0.87
    assert boltz_conf["ptm"] == 0.81
    assert "model" not in boltz_conf  # non-numeric values are dropped

    of_conf = read_confidence(structures["openfold"])
    assert of_conf["ranking_score"] == 0.74


def test_structure_text_and_format(tmp_path):
    _make_output_tree(tmp_path)
    structures = {s.stage: s for s in find_structures(tmp_path / "sequences" / "mutant_A")}
    assert structure_format(structures["boltz"].path) == "cif"
    assert read_structure_text(structures["boltz"].path).startswith("data_x")


def test_read_benchmarks(tmp_path):
    _make_output_tree(tmp_path)
    benches = read_benchmarks(tmp_path)

    assert set(benches) == {"boltz", "msa"}
    boltz = benches["boltz"]
    assert boltz.n_jobs == 2
    assert boltz.total_s == 300.0
    assert boltz.mean_s == 150.0
    assert boltz.max_rss_mb == 10000
    assert abs(boltz.node_hours - 300.0 / 3600) < 1e-9


def test_empty_tree_is_graceful(tmp_path):
    assert list_result_sequences(tmp_path) == []
    assert read_benchmarks(tmp_path) == {}
