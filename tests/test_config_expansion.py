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


class TestNonStrictExpansion:
    """The webapp seeds a session from the template before PROTFORGE_ROOT is
    necessarily exported, so it needs expansion that keeps unresolved values
    rather than raising. The strict pass at launch still rejects them."""

    def test_unset_var_is_left_intact(self):
        cfg = {"output": {"parent_dir": "${NOPE_NOT_SET}/outputs"}}
        assert expand_config(cfg, strict=False) == cfg

    def test_resolvable_values_still_expand_alongside_unresolvable_ones(self, monkeypatch):
        monkeypatch.setenv("PROTFORGE_ROOT", "/n/lab/me")
        out = expand_config(
            {"a": "${PROTFORGE_ROOT}/outputs", "b": "${NOPE_NOT_SET}/x", "c": "/literal"},
            strict=False,
        )
        assert out == {"a": "/n/lab/me/outputs", "b": "${NOPE_NOT_SET}/x", "c": "/literal"}

    def test_non_strict_recurses_through_lists(self, monkeypatch):
        monkeypatch.delenv("PROTFORGE_ROOT", raising=False)
        out = expand_config({"binds": ["${PROTFORGE_ROOT}/a", "/literal"]}, strict=False)
        assert out["binds"] == ["${PROTFORGE_ROOT}/a", "/literal"]

    def test_strict_remains_the_default(self):
        with pytest.raises(KeyError):
            expand_config({"p": "${NOPE_NOT_SET}/x"})


class TestKempnerTemplate:
    """The shipped template must stay valid and expandable — it's what new users
    copy, so a typo here breaks every fresh install."""

    template = REPO_ROOT / "config.kempner.template.yaml"

    def test_template_exists_and_is_valid_yaml(self):
        assert self.template.is_file()
        assert isinstance(yaml.safe_load(self.template.read_text()), dict)

    def _expanded(self, monkeypatch, root="/n/lab/me", assets="/n/lab/shared"):
        monkeypatch.setenv("PROTFORGE_ROOT", root)
        monkeypatch.setenv("PROTFORGE_ASSETS", assets)
        return expand_config(yaml.safe_load(self.template.read_text()))

    def test_template_expands_with_both_vars_set(self, monkeypatch):
        cfg = self._expanded(monkeypatch)

        assert cfg["output"]["parent_dir"] == "/n/lab/me/outputs"
        assert cfg["containers"]["esmc"] == "/n/lab/shared/sifs/esm.sif"
        # esmc and esmfold intentionally share one image.
        assert cfg["containers"]["esmfold"] == cfg["containers"]["esmc"]
        # Shared cluster paths must NOT be parameterized on either variable.
        assert cfg["msa"]["mmseq2_db"].startswith(
            "/n/holylfs06/LABS/kempner_shared/"
        )
        assert cfg["boltz"]["cache_dir"].startswith(
            "/n/holylfs06/LABS/kempner_shared/"
        )

    def test_assets_and_workspace_are_independent(self, monkeypatch):
        """The point of the split: read-only images and weights can live in a
        shared copy while inputs, outputs and logs stay per-user."""
        cfg = self._expanded(monkeypatch)

        for key in ("colabfold", "boltz", "esmc", "esmfold", "openfold"):
            assert cfg["containers"][key].startswith("/n/lab/shared/")
        for section in ("esmc", "esmfold", "openfold"):
            assert cfg[section]["cache_dir"].startswith("/n/lab/shared/")

        assert cfg["input"]["fasta_dir"].startswith("/n/lab/me/")
        assert cfg["output"]["parent_dir"].startswith("/n/lab/me/")
        assert cfg["slurm"]["log_dir"].startswith("/n/lab/me/")

    def test_one_workspace_for_everything_still_works(self, monkeypatch):
        """Users who build their own images set both variables to the same dir."""
        cfg = self._expanded(monkeypatch, root="/n/lab/me", assets="/n/lab/me")
        assert cfg["containers"]["boltz"] == "/n/lab/me/sifs/boltz.sif"
        assert cfg["output"]["parent_dir"] == "/n/lab/me/outputs"

    def test_template_fails_clearly_without_protforge_root(self, monkeypatch):
        monkeypatch.delenv("PROTFORGE_ROOT", raising=False)
        monkeypatch.setenv("PROTFORGE_ASSETS", "/n/lab/shared")
        with pytest.raises(KeyError, match="PROTFORGE_ROOT"):
            expand_config(yaml.safe_load(self.template.read_text()))

    def test_template_fails_clearly_without_protforge_assets(self, monkeypatch):
        monkeypatch.setenv("PROTFORGE_ROOT", "/n/lab/me")
        monkeypatch.delenv("PROTFORGE_ASSETS", raising=False)
        with pytest.raises(KeyError, match="PROTFORGE_ASSETS"):
            expand_config(yaml.safe_load(self.template.read_text()))

    def test_template_leaves_account_and_email_as_placeholders(self):
        """These are the fields we tell the user to edit; keep them un-runnable
        so a forgotten edit fails fast rather than mailing the wrong person."""
        cfg = yaml.safe_load(self.template.read_text())
        assert cfg["slurm"]["account"].startswith("<")
        assert cfg["slurm"]["email"].startswith("<")
