"""Tests for the preflight checks in pipeline_ops.

`slurm.account` reaches every rule's slurm_account and `slurm.email` reaches
sbatch --mail-user, but neither used to be validated — a config carrying
another user's identity, or the shipped template's placeholders, launched
without complaint. These lock the checks in place.
"""

from pipeline_ops import (
    check_launch_inputs,
    check_stage_assets,
    check_unexpanded_paths,
)
from snake_helpers import slurm_identity_errors

GOOD_SLURM = {"account": "kempner_x_lab", "email": "me@uni.edu"}


def _joined(errors: list[str]) -> str:
    return " | ".join(errors)


def _sif(tmp_path, name="fake.sif"):
    path = tmp_path / name
    path.touch()
    return str(path)


class TestSlurmIdentity:
    def test_valid_account_and_email_pass(self):
        assert slurm_identity_errors({"slurm": GOOD_SLURM}) == []

    def test_missing_slurm_block_is_an_error(self):
        errors = slurm_identity_errors({})
        assert len(errors) == 1
        assert "slurm.account" in errors[0]

    def test_empty_account_is_an_error(self):
        errors = slurm_identity_errors({"slurm": {"account": "  ", "email": "me@uni.edu"}})
        assert "slurm.account is not set" in _joined(errors)

    def test_placeholder_account_is_an_error(self):
        errors = slurm_identity_errors(
            {"slurm": {"account": "<YOUR_KEMPNER_ACCOUNT>", "email": "me@uni.edu"}}
        )
        assert "placeholder" in _joined(errors)

    def test_placeholder_email_is_an_error(self):
        errors = slurm_identity_errors(
            {"slurm": {"account": "kempner_x_lab", "email": "<YOUR_EMAIL>"}}
        )
        assert "slurm.email is still the template placeholder" in _joined(errors)

    def test_empty_email_is_allowed(self):
        """Notifications are optional; an empty address just disables them."""
        assert slurm_identity_errors({"slurm": {"account": "kempner_x_lab", "email": ""}}) == []

    def test_missing_email_key_is_allowed(self):
        assert slurm_identity_errors({"slurm": {"account": "kempner_x_lab"}}) == []

    def test_malformed_email_is_an_error(self):
        errors = slurm_identity_errors({"slurm": {"account": "kempner_x_lab", "email": "nope"}})
        assert "does not look like an address" in _joined(errors)

    def test_null_slurm_block_does_not_crash(self):
        errors = slurm_identity_errors({"slurm": None})
        assert "slurm.account" in _joined(errors)


class TestRequireAccountOff:
    """The Snakefile path: a local run needs no account, but a placeholder is
    still wrong wherever it appears."""

    def test_missing_account_is_allowed(self):
        assert slurm_identity_errors({}, require_account=False) == []

    def test_placeholder_account_still_rejected(self):
        errors = slurm_identity_errors(
            {"slurm": {"account": "<YOUR_KEMPNER_ACCOUNT>"}}, require_account=False
        )
        assert "placeholder" in _joined(errors)

    def test_placeholder_email_still_rejected(self):
        errors = slurm_identity_errors(
            {"slurm": {"email": "<YOUR_EMAIL>"}}, require_account=False
        )
        assert "placeholder" in _joined(errors)


class TestUnexpandedPaths:
    def test_literal_paths_pass(self):
        assert check_unexpanded_paths({"output": {"parent_dir": "/n/lab/me/outputs"}}) == []

    def test_expandable_var_passes(self, monkeypatch):
        monkeypatch.setenv("PROTFORGE_ROOT", "/n/lab/me")
        assert check_unexpanded_paths({"output": {"parent_dir": "${PROTFORGE_ROOT}/out"}}) == []

    def test_unset_var_is_reported_with_key_and_var_name(self, monkeypatch):
        monkeypatch.delenv("PROTFORGE_ROOT", raising=False)
        errors = check_unexpanded_paths({"output": {"parent_dir": "${PROTFORGE_ROOT}/out"}})
        assert len(errors) == 1
        assert "output.parent_dir" in errors[0]
        assert "PROTFORGE_ROOT" in errors[0]


class TestStageAssets:
    """A seeded config points at SIFs and weight caches that only exist after
    containers/build.sh and download_models.py have run."""

    def test_no_enabled_stages_needs_nothing(self):
        assert check_stage_assets({"pipeline": {}}) == []

    def test_missing_container_is_reported_with_build_hint(self, tmp_path):
        cfg = {
            "pipeline": {"msa": True},
            "containers": {"colabfold": str(tmp_path / "absent.sif")},
        }
        errors = check_stage_assets(cfg)
        assert "container image not found" in _joined(errors)
        assert "containers/build.sh" in _joined(errors)

    def test_unset_container_is_reported(self):
        errors = check_stage_assets({"pipeline": {"boltz": True}, "containers": {}})
        assert "containers.boltz is not set" in _joined(errors)

    def test_existing_container_passes(self, tmp_path):
        cfg = {"pipeline": {"msa": True}, "containers": {"colabfold": _sif(tmp_path)}}
        assert check_stage_assets(cfg) == []

    def test_shared_gpu_image_satisfies_a_stage(self, tmp_path):
        cfg = {"pipeline": {"boltz": True}, "containers": {"gpu": _sif(tmp_path)}}
        assert check_stage_assets(cfg) == []

    def test_disabled_stage_assets_are_not_checked(self):
        cfg = {"pipeline": {"msa": True, "openfold": False}, "containers": {"gpu": ""}}
        errors = check_stage_assets(cfg)
        assert "openfold" not in _joined(errors)

    def test_missing_model_cache_is_reported_with_download_hint(self, tmp_path):
        cfg = {
            "pipeline": {"esmc": True},
            "containers": {"esmc": _sif(tmp_path)},
            "esmc": {"cache_dir": str(tmp_path / "no_such_cache")},
        }
        errors = check_stage_assets(cfg)
        assert "model cache not found" in _joined(errors)
        assert "download_models.py" in _joined(errors)

    def test_existing_cache_passes(self, tmp_path):
        cache = tmp_path / "hf"
        cache.mkdir()
        cfg = {
            "pipeline": {"esmc": True},
            "containers": {"esmc": _sif(tmp_path)},
            "esmc": {"cache_dir": str(cache)},
        }
        assert check_stage_assets(cfg) == []

    def test_stages_without_a_user_cache_skip_the_cache_check(self, tmp_path):
        """MSA databases and the Boltz checkpoint are shared cluster paths."""
        cfg = {
            "pipeline": {"msa": True, "boltz": True},
            "containers": {"gpu": _sif(tmp_path)},
        }
        assert check_stage_assets(cfg) == []


class TestCheckLaunchInputs:
    """The identity checks must survive every early return in the input logic."""

    def test_identity_errors_surface_when_input_dir_is_missing(self, tmp_path):
        cfg = {
            "pipeline": {"msa": True},
            "input": {"fasta_dir": str(tmp_path / "nope")},
            "slurm": {"account": "<YOUR_KEMPNER_ACCOUNT>"},
        }
        joined = _joined(check_launch_inputs(cfg)["errors"])
        assert "placeholder" in joined
        assert "does not exist" in joined

    def test_identity_errors_surface_when_fasta_dir_unset(self):
        cfg = {"pipeline": {"msa": True}, "input": {}, "slurm": {}}
        joined = _joined(check_launch_inputs(cfg)["errors"])
        assert "slurm.account" in joined
        assert "input.fasta_dir is not set" in joined

    def test_identity_errors_surface_on_the_msa_off_path(self, tmp_path):
        cfg = {
            "pipeline": {"msa": False, "boltz": True},
            "input": {},
            "slurm": {"account": ""},
        }
        joined = _joined(check_launch_inputs(cfg)["errors"])
        assert "slurm.account" in joined
        assert "input.yaml_dir is not set" in joined

    def test_missing_container_blocks_an_otherwise_valid_run(self, tmp_path):
        """The gap a fresh clone hits: config is correct, images aren't built."""
        fasta_dir = tmp_path / "fastas"
        fasta_dir.mkdir()
        (fasta_dir / "a.fasta").write_text(">a\nMKV\n")
        cfg = {
            "pipeline": {"msa": True},
            "input": {"fasta_dir": str(fasta_dir)},
            "containers": {"colabfold": str(tmp_path / "sifs" / "msa.sif")},
            "slurm": GOOD_SLURM,
        }
        assert "container image not found" in _joined(check_launch_inputs(cfg)["errors"])

    def test_valid_run_has_no_errors(self, tmp_path):
        fasta_dir = tmp_path / "fastas"
        fasta_dir.mkdir()
        (fasta_dir / "a.fasta").write_text(">a\nMKV\n")
        cfg = {
            "pipeline": {"msa": True},
            "input": {"fasta_dir": str(fasta_dir)},
            "output": {"parent_dir": str(tmp_path / "out")},
            "containers": {"colabfold": _sif(tmp_path)},
            "slurm": GOOD_SLURM,
        }
        assert check_launch_inputs(cfg)["errors"] == []
