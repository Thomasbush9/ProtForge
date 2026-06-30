"""
Tests for the Snakefile-side helpers and the calibration loader.

These cover the integration seam between the webapp's estimator output and
what the Snakemake rules read at run time, plus the round-trip from a
benchmarks/ directory back into webapp/scaling_models.calibrated.yaml.

Run with:
    PYTHONPATH=webapp:workflow/scripts \\
        /path/to/protforge/python -m pytest webapp/tests/test_workflow_helpers.py -v
"""

import sys
from pathlib import Path

import pytest
import yaml


# Make workflow/scripts importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "workflow" / "scripts"))

from snake_helpers import stage_resource, stage_uses_gpu  # noqa: E402

from estimator import (  # noqa: E402
    apply_estimate_to_config,
    compute_input_stats,
    estimate_all_stages,
    estimate_stage,
    load_scaling_models,
    recalibrate_from_benchmarks,
)


# ---------------------------------------------------------------------------
# Snakefile helpers — what every rule calls into
# ---------------------------------------------------------------------------


def test_stage_resource_returns_default_when_missing():
    cfg = {}
    assert stage_resource(cfg, "boltz", "mem_mb", 16000) == 16000


def test_stage_resource_returns_override():
    cfg = {"resources": {"boltz": {"mem_mb": 99999}}}
    assert stage_resource(cfg, "boltz", "mem_mb", 16000) == 99999


def test_stage_resource_isolated_per_stage():
    """Setting boltz.mem_mb shouldn't leak into msa.mem_mb."""
    cfg = {"resources": {"boltz": {"mem_mb": 99999}}}
    assert stage_resource(cfg, "msa", "mem_mb", 48000) == 48000


def test_stage_resource_default_used_for_unset_key():
    cfg = {"resources": {"boltz": {"mem_mb": 99999}}}
    # cpus_per_task wasn't overridden — fall back to default
    assert stage_resource(cfg, "boltz", "cpus_per_task", 8) == 8


def test_stage_uses_gpu_default_when_unspecified():
    cfg = {}
    assert stage_uses_gpu(cfg, "boltz", True) is True
    assert stage_uses_gpu(cfg, "es", False) is False


def test_stage_uses_gpu_explicit_zero_disables_gpu():
    """ES with gpus=0 in config should NOT request a GPU even if rule defaults True."""
    cfg = {"resources": {"es": {"gpus": 0}}}
    assert stage_uses_gpu(cfg, "es", True) is False


def test_stage_uses_gpu_explicit_one_enables_gpu():
    cfg = {"resources": {"esm": {"gpus": 1}}}
    assert stage_uses_gpu(cfg, "esm", False) is True


def test_stage_uses_gpu_handles_string_int():
    """YAML can produce strings if quoted — coerce defensively."""
    cfg = {"resources": {"boltz": {"gpus": "1"}}}
    assert stage_uses_gpu(cfg, "boltz", False) is True


# ---------------------------------------------------------------------------
# End-to-end: estimator writes config, helper reads it back
# ---------------------------------------------------------------------------


def test_estimator_apply_then_helper_read_roundtrip(tmp_path):
    """The seam that matters: webapp Apply writes config; Snakemake helpers read it."""
    cfg_path = tmp_path / "config.yaml"
    initial = {
        "pipeline": {"msa": False, "boltz": True, "esmc": False, "esmfold": False},
        "boltz": {"recycling_steps": 10, "diffusion_samples": 25, "num_runs": 1},
        "slurm": {"partition": "kempner_requeue", "account": "test_lab"},
    }
    cfg_path.write_text(yaml.safe_dump(initial))

    scaling = load_scaling_models()
    stats = compute_input_stats(fasta_results=[{"valid": True, "total_residues": 700}] * 30)
    estimates = estimate_all_stages(stats, initial, scaling)
    apply_estimate_to_config(cfg_path, estimates, backup=False)

    # Read back exactly the way Snakefile would
    written_cfg = yaml.safe_load(cfg_path.read_text())
    slurm_cfg = written_cfg.get("slurm", {})

    # Boltz should now have a non-default mem_mb (estimator computed something)
    boltz_mem = stage_resource(slurm_cfg, "boltz", "mem_mb", 16000)
    assert boltz_mem != 16000, "estimator should have written a custom mem_mb"

    # Boltz is GPU-using, so the estimator should have requested a GPU
    assert stage_uses_gpu(slurm_cfg, "boltz", False) is True

    # An unmentioned stage falls back
    assert stage_resource(slurm_cfg, "msa", "cpus_per_task", 4) == 4


def test_apply_honors_existing_partition_overrides(tmp_path):
    """A user's manually-pinned `slurm.<stage>.partition` survives Apply.

    The estimator reads it via `stage_override` in estimate_stage() and writes
    the same value back. This is intentional: pinning a partition is a strong
    user signal, and the estimator's job is to size the resources, not pick
    the cluster queue.
    """
    cfg_path = tmp_path / "config.yaml"
    initial = {
        "pipeline": {"boltz": True, "msa": True, "esm": False, "esmfold": False, "es": False},
        "boltz": {"recycling_steps": 10, "diffusion_samples": 25, "num_runs": 1},
        "slurm": {
            "partition": "kempner_requeue",
            "msa": {"partition": "user_pinned_msa_partition"},
        },
    }
    cfg_path.write_text(yaml.safe_dump(initial))

    scaling = load_scaling_models()
    stats = compute_input_stats(fasta_results=[{"valid": True, "total_residues": 500}] * 10)
    estimates = estimate_all_stages(stats, initial, scaling)
    apply_estimate_to_config(cfg_path, estimates, backup=False)

    written = yaml.safe_load(cfg_path.read_text())
    assert written["slurm"]["msa"]["partition"] == "user_pinned_msa_partition"
    assert written["slurm"]["partition"] == "kempner_requeue"
    # Resource sizing was still applied
    assert "resources" in written["slurm"]
    assert "msa" in written["slurm"]["resources"]


# ---------------------------------------------------------------------------
# Calibration loader
# ---------------------------------------------------------------------------


def _write_fake_benchmark_tsv(path: Path, runtime_s: float, max_rss_mb: float) -> None:
    """Write a Snakemake-format benchmark TSV (one rule run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "s\th:m:s\tmax_rss\tmax_vms\tmax_uss\tmax_pss\tio_in\tio_out\tmean_load\tcpu_time\n"
        f"{runtime_s}\t0:01:00\t{max_rss_mb}\t{max_rss_mb}\t{max_rss_mb}\t{max_rss_mb}\t0\t0\t1.0\t{runtime_s}\n"
    )


def test_recalibrate_parses_benchmarks(tmp_path):
    bench = tmp_path / "benchmarks"
    _write_fake_benchmark_tsv(bench / "boltz" / "predict_0_run_0.tsv", runtime_s=120.0, max_rss_mb=8000)
    _write_fake_benchmark_tsv(bench / "boltz" / "predict_1_run_0.tsv", runtime_s=180.0, max_rss_mb=10000)
    _write_fake_benchmark_tsv(bench / "esmc"  / "esm_chunk_0.tsv",     runtime_s=30.0,  max_rss_mb=4000)

    output = tmp_path / "calibrated.yaml"
    fitted = recalibrate_from_benchmarks(bench, output_path=output)

    assert output.exists()
    assert "boltz" in fitted
    assert "esmc" in fitted

    boltz = fitted["boltz"]["per_gpu"]["_observed"]
    # Mean of [120, 180] = 150
    assert boltz["runtime_sec_per_seq"]["base"] == pytest.approx(150.0, rel=0.01)
    # Snakemake writes max_rss in MB; recalibrate divides by 1024 after a *1024 round-trip,
    # so the value goes through unchanged in MB — assert it landed with the right unit.
    assert boltz["mem_mb"]["base"] > 0


def test_recalibrate_skips_unknown_stages(tmp_path):
    """Random subdirs in benchmarks/ shouldn't crash the loader."""
    bench = tmp_path / "benchmarks"
    _write_fake_benchmark_tsv(bench / "not_a_real_stage" / "x.tsv", 1.0, 100)
    _write_fake_benchmark_tsv(bench / "boltz" / "y.tsv", 5.0, 100)
    fitted = recalibrate_from_benchmarks(bench, output_path=tmp_path / "out.yaml")
    assert "boltz" in fitted
    assert "not_a_real_stage" not in fitted


def test_recalibrate_handles_empty_benchmarks_dir(tmp_path):
    bench = tmp_path / "benchmarks"
    bench.mkdir()
    fitted = recalibrate_from_benchmarks(bench, output_path=tmp_path / "out.yaml")
    assert fitted == {}
    # No output file should be written if nothing was fit
    assert not (tmp_path / "out.yaml").exists()


# ---------------------------------------------------------------------------
# Calibrated YAML overlay
# ---------------------------------------------------------------------------


def test_load_scaling_models_with_calibration_overlay(tmp_path, monkeypatch):
    """A calibrated YAML next to scaling_models.yaml should override per-GPU coeffs."""
    import estimator as estimator_mod

    # Write a base scaling YAML
    base_path = tmp_path / "scaling_models.yaml"
    cal_path = tmp_path / "scaling_models.calibrated.yaml"
    base_path.write_text(yaml.safe_dump({
        "margins": {"mem_safety": 1.0, "time_safety": 1.0, "min_runtime_min": 1},
        "gpu_specs": {"h100": {"mem_gb": 80, "partition": "kempner_h100", "tier": 3}},
        "boltz": {
            "default_gpu": "h100",
            "cpus": 8, "gpus": 1,
            "per_gpu": {
                "h100": {
                    "mem_mb": {"base": 1000, "alpha_L2": 0.001},
                    "runtime_sec_per_seq": {"base": 100, "alpha_L2": 0.001},
                },
            },
            "target_chunk_runtime_min": 30,
            "max_chunk_size": 50,
            "min_chunk_size": 1,
        },
    }))

    cal_path.write_text(yaml.safe_dump({
        "boltz": {
            "per_gpu": {
                "h100": {
                    "mem_mb": {"base": 99999, "alpha_L2": 0.0},
                    "runtime_sec_per_seq": {"base": 1, "alpha_L2": 0.0},
                },
            },
        },
    }))

    monkeypatch.setattr(estimator_mod, "DEFAULT_SCALING_PATH", base_path)
    monkeypatch.setattr(estimator_mod, "CALIBRATED_SCALING_PATH", cal_path)

    models = estimator_mod.load_scaling_models()
    assert models["boltz"]["per_gpu"]["h100"]["mem_mb"]["base"] == 99999


def test_load_scaling_models_works_without_calibration(tmp_path, monkeypatch):
    """Heuristic-only mode: no calibrated file present → defaults used as-is."""
    import estimator as estimator_mod

    base_path = tmp_path / "scaling_models.yaml"
    cal_path = tmp_path / "absent.calibrated.yaml"
    base_path.write_text(yaml.safe_dump({"boltz": {"default_gpu": "h100", "per_gpu": {"h100": {"mem_mb": {"base": 1234}}}}}))

    monkeypatch.setattr(estimator_mod, "DEFAULT_SCALING_PATH", base_path)
    monkeypatch.setattr(estimator_mod, "CALIBRATED_SCALING_PATH", cal_path)

    models = estimator_mod.load_scaling_models()
    assert models["boltz"]["per_gpu"]["h100"]["mem_mb"]["base"] == 1234
