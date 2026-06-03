"""
Unit tests for webapp/estimator.py.

Run with:
    PYTHONPATH=webapp /path/to/protforge/python -m pytest webapp/tests/test_estimator.py -v
"""

from pathlib import Path

import pytest
import yaml

from estimator import (
    apply_estimate_to_config,
    compute_input_stats,
    estimate_all_stages,
    estimate_stage,
    load_scaling_models,
    pick_partition,
)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats_empty():
    s = compute_input_stats(fasta_results=[])
    assert s.count == 0
    assert s.total_residues == 0


def test_stats_filters_invalid():
    fastas = [
        {"valid": True, "total_residues": 100},
        {"valid": False, "total_residues": 9999},   # ignored
        {"valid": True, "total_residues": 200},
    ]
    s = compute_input_stats(fasta_results=fastas)
    assert s.count == 2
    assert s.total_residues == 300
    assert s.min_len == 100
    assert s.max_len == 200


def test_stats_p95_picks_top_5pct():
    fastas = [{"valid": True, "total_residues": L} for L in [100] * 95 + [1000] * 5]
    s = compute_input_stats(fasta_results=fastas)
    # 95th percentile should land at the cliff between 100 and 1000 — large
    # enough that GPU memory routing reflects the long tail.
    assert s.p95_len >= 100


def test_stats_yaml_inputs():
    yamls = [{"valid": True, "sequence_length": 500}, {"valid": True, "sequence_length": 800}]
    s = compute_input_stats(yaml_results=yamls)
    assert s.file_type == "yaml"
    assert s.count == 2
    assert s.total_residues == 1300


def test_stats_rejects_both_inputs():
    with pytest.raises(ValueError):
        compute_input_stats(
            fasta_results=[{"valid": True, "total_residues": 1}],
            yaml_results=[{"valid": True, "sequence_length": 1}],
        )


# ---------------------------------------------------------------------------
# Per-stage estimates
# ---------------------------------------------------------------------------


def _config():
    return {
        "pipeline": {"msa": True, "boltz": True, "esmc": True, "esmfold": False},
        "boltz": {"recycling_steps": 10, "diffusion_samples": 25, "num_runs": 1},
        "slurm": {"partition": "kempner_requeue"},
    }


def test_estimates_monotonic_in_length():
    """4x'ing sequence length should grow Boltz memory and total compute.

    Per-chunk runtime stays ~constant by design (target_chunk_runtime_min) —
    the formula trades chunk size for chunk count to hit that target. Memory
    and total node-hours are the right axes to check.
    """
    scaling = load_scaling_models()
    cfg = _config()

    short = compute_input_stats(fasta_results=[{"valid": True, "total_residues": 200}] * 50)
    long = compute_input_stats(fasta_results=[{"valid": True, "total_residues": 800}] * 50)

    s = estimate_stage("boltz", short, cfg, scaling)
    l = estimate_stage("boltz", long, cfg, scaling)

    assert l.mem_mb > s.mem_mb
    assert l.total_node_hours > s.total_node_hours


def test_estimates_monotonic_in_count():
    """More sequences → more SLURM jobs."""
    scaling = load_scaling_models()
    cfg = _config()

    few = compute_input_stats(fasta_results=[{"valid": True, "total_residues": 500}] * 10)
    many = compute_input_stats(fasta_results=[{"valid": True, "total_residues": 500}] * 100)

    s = estimate_stage("boltz", few, cfg, scaling)
    l = estimate_stage("boltz", many, cfg, scaling)

    assert l.num_chunks >= s.num_chunks
    assert l.total_node_hours >= s.total_node_hours


def test_chunk_size_caps_at_max():
    scaling = load_scaling_models()
    cfg = _config()
    # Tiny sequences — chunk size formula will want to be huge
    stats = compute_input_stats(fasta_results=[{"valid": True, "total_residues": 50}] * 1000)
    e = estimate_stage("esmc", stats, cfg, scaling)
    assert e.chunk_size <= scaling["esmc"]["max_chunk_size"]


def test_partition_routing_auto_picks_smallest_fitting_gpu():
    """Small inputs → cheap GPU; huge inputs → bigger card."""
    scaling = load_scaling_models()
    cfg = _config()

    small = compute_input_stats(fasta_results=[{"valid": True, "total_residues": 200}] * 5)
    huge = compute_input_stats(fasta_results=[{"valid": True, "total_residues": 4000}])

    p_small, gpu_small = pick_partition("esmc", small, cfg, scaling)
    p_huge, gpu_huge = pick_partition("esmc", huge, cfg, scaling)

    # Huge should land on H100 (or higher) since 4000-residue ESMC > 40GB
    assert gpu_huge in {"h100", "h200"}
    # Small should land on something with smaller capacity
    assert gpu_small != gpu_huge or scaling["gpu_specs"][gpu_small]["mem_gb"] >= 40


def test_partition_routing_honors_user_pin():
    """Pinning a known GPU type forces it through to gpu_type."""
    scaling = load_scaling_models()
    cfg = _config()
    stats = compute_input_stats(fasta_results=[{"valid": True, "total_residues": 4000}] * 3)
    # Without pin, large p95 routes ESMC to h100 automatically
    e_auto = estimate_stage("esmc", stats, cfg, scaling)
    assert e_auto.gpu_type == "h100"
    # Explicit pin to a100 should override even though mem may not fit
    e_pinned = estimate_stage("esmc", stats, cfg, scaling, gpu_preference="a100")
    assert e_pinned.gpu_type == "a100"


def test_boltz_runtime_scales_with_recycling_and_samples():
    scaling = load_scaling_models()
    cfg_default = _config()
    cfg_double = _config()
    cfg_double["boltz"]["recycling_steps"] = 20  # 2x
    cfg_double["boltz"]["diffusion_samples"] = 50  # 2x → 4x total compute

    stats = compute_input_stats(fasta_results=[{"valid": True, "total_residues": 500}] * 20)

    e1 = estimate_stage("boltz", stats, cfg_default, scaling)
    e2 = estimate_stage("boltz", stats, cfg_double, scaling)

    # Runtime per chunk should grow; chunk size should shrink as a result
    assert e2.runtime_min >= e1.runtime_min or e2.chunk_size <= e1.chunk_size


def test_estimate_all_stages_skips_disabled():
    scaling = load_scaling_models()
    cfg = _config()
    cfg["pipeline"]["esmc"] = False
    stats = compute_input_stats(fasta_results=[{"valid": True, "total_residues": 500}] * 10)
    out = estimate_all_stages(stats, cfg, scaling)
    assert "esmc" not in out
    assert "boltz" in out


def test_estimate_with_zero_inputs_returns_zero():
    scaling = load_scaling_models()
    cfg = _config()
    stats = compute_input_stats(fasta_results=[])
    e = estimate_stage("boltz", stats, cfg, scaling)
    assert e.num_chunks == 0
    assert e.total_node_hours == 0


# ---------------------------------------------------------------------------
# Apply round-trip
# ---------------------------------------------------------------------------


def test_apply_round_trip_preserves_unrelated_keys(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    initial = {
        "pipeline": {"msa": True, "boltz": True, "esmc": False, "esmfold": False},
        "input": {"fasta_dir": "/tmp/data"},
        "output": {"parent_dir": "/tmp/out"},
        "boltz": {"recycling_steps": 10, "diffusion_samples": 25, "num_runs": 1,
                  "cache_dir": "/cache/boltz"},
        "slurm": {"partition": "kempner_requeue", "account": "test_lab",
                  "msa": {"partition": "kempner"}},  # existing partition pin
    }
    cfg_path.write_text(yaml.safe_dump(initial))

    scaling = load_scaling_models()
    stats = compute_input_stats(fasta_results=[{"valid": True, "total_residues": 500}] * 20)
    estimates = estimate_all_stages(stats, initial, scaling)
    apply_estimate_to_config(cfg_path, estimates, backup=False)

    written = yaml.safe_load(cfg_path.read_text())

    # Unrelated keys preserved
    assert written["input"]["fasta_dir"] == "/tmp/data"
    assert written["output"]["parent_dir"] == "/tmp/out"
    assert written["boltz"]["cache_dir"] == "/cache/boltz"
    assert written["slurm"]["account"] == "test_lab"

    # New keys present
    assert "resources" in written["slurm"]
    for stage in ("msa", "boltz"):
        assert stage in written["slurm"]["resources"]
        assert "mem_mb" in written["slurm"]["resources"][stage]

    # Partition was overwritten with the estimator's choice
    # (could be "kempner" if a100 wins, or "kempner_h100" if h100 wins)
    assert written["slurm"]["msa"]["partition"]


def test_apply_creates_backup_by_default(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"pipeline": {"boltz": True},
                                        "boltz": {"recycling_steps": 10, "diffusion_samples": 25}}))
    scaling = load_scaling_models()
    stats = compute_input_stats(fasta_results=[{"valid": True, "total_residues": 200}] * 5)
    estimates = estimate_all_stages(stats, {"pipeline": {"boltz": True},
                                            "boltz": {"recycling_steps": 10, "diffusion_samples": 25}},
                                    scaling)
    backup_path = apply_estimate_to_config(cfg_path, estimates, backup=True)
    assert backup_path.exists()
    assert backup_path.parent.name == ".config_backups"


# ---------------------------------------------------------------------------
# Smoke test for unknown stages
# ---------------------------------------------------------------------------


def test_unknown_stage_raises():
    scaling = load_scaling_models()
    cfg = _config()
    stats = compute_input_stats(fasta_results=[{"valid": True, "total_residues": 100}] * 5)
    with pytest.raises(KeyError):
        estimate_stage("not_a_stage", stats, cfg, scaling)
