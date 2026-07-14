"""Tests for expand_config() in workflow/scripts/snake_helpers.py, and for the
shipped Kempner template staying loadable with it."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflow" / "scripts"))
from snake_helpers import expand_config

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestExpandConfig:
    def test_expands_var_in_nested_paths(self, monkeypatch):
        monkeypatch.setenv("PROTFORGE_ROOT", "/n/lab/me")
        cfg = {"output": {"parent_dir": "${PROTFORGE_ROOT}/outputs"}}
        assert expand_config(cfg)["output"]["parent_dir"] == "/n/lab/me/outputs"

    def test_expands_bare_dollar_form(self, monkeypatch):
        monkeypatch.setenv("PROTFORGE_ROOT", "/n/lab/me")
        assert expand_config({"p": "$PROTFORGE_ROOT/sifs/esm.sif"})["p"] == (
            "/n/lab/me/sifs/esm.sif"
        )

    def test_expands_inside_lists(self, monkeypatch):
        monkeypatch.setenv("PROTFORGE_ROOT", "/n/lab/me")
        out = expand_config({"binds": ["${PROTFORGE_ROOT}/a", "/literal/b"]})
        assert out["binds"] == ["/n/lab/me/a", "/literal/b"]

    def test_expands_tilde(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/me")
        assert expand_config({"p": "~/cache"})["p"] == "/home/me/cache"

    def test_literal_paths_untouched(self):
        """Backward compatibility: an existing config with plain paths is a no-op."""
        cfg = {
            "msa": {"mmseq2_db": "/n/holylfs06/LABS/kempner_shared/x"},
            "boltz": {"recycling_steps": 10, "delete_msa_after_processing": False},
            "esmc": {"models": ["600M"]},
        }
        assert expand_config(cfg) == cfg

    def test_non_string_leaves_preserve_type(self):
        cfg = {"a": 10, "b": True, "c": None, "d": 1.5}
        assert expand_config(cfg) == cfg

    def test_unset_var_raises_naming_key_and_var(self):
        """A missing env var must fail loudly at parse time, not build a bad path."""
        with pytest.raises(KeyError) as exc:
            expand_config({"output": {"parent_dir": "${NOPE_NOT_SET}/outputs"}})
        msg = str(exc.value)
        assert "output.parent_dir" in msg
        assert "NOPE_NOT_SET" in msg


class TestKempnerTemplate:
    """The shipped template must stay valid and expandable — it's what new users
    copy, so a typo here breaks every fresh install."""

    template = REPO_ROOT / "config.kempner.template.yaml"

    def test_template_exists_and_is_valid_yaml(self):
        assert self.template.is_file()
        assert isinstance(yaml.safe_load(self.template.read_text()), dict)

    def test_template_expands_with_protforge_root_set(self, monkeypatch):
        monkeypatch.setenv("PROTFORGE_ROOT", "/n/lab/me")
        cfg = expand_config(yaml.safe_load(self.template.read_text()))

        assert cfg["output"]["parent_dir"] == "/n/lab/me/outputs"
        assert cfg["containers"]["esmc"] == "/n/lab/me/sifs/esm.sif"
        # esmc and esmfold intentionally share one image.
        assert cfg["containers"]["esmfold"] == cfg["containers"]["esmc"]
        # Shared cluster paths must NOT be parameterized on the user's workspace.
        assert cfg["msa"]["mmseq2_db"].startswith(
            "/n/holylfs06/LABS/kempner_shared/"
        )
        assert cfg["boltz"]["cache_dir"].startswith(
            "/n/holylfs06/LABS/kempner_shared/"
        )

    def test_template_fails_clearly_without_protforge_root(self, monkeypatch):
        monkeypatch.delenv("PROTFORGE_ROOT", raising=False)
        with pytest.raises(KeyError, match="PROTFORGE_ROOT"):
            expand_config(yaml.safe_load(self.template.read_text()))

    def test_template_leaves_account_and_email_as_placeholders(self):
        """These are the fields we tell the user to edit; keep them un-runnable
        so a forgotten edit fails fast rather than mailing the wrong person."""
        cfg = yaml.safe_load(self.template.read_text())
        assert cfg["slurm"]["account"].startswith("<")
        assert cfg["slurm"]["email"].startswith("<")
