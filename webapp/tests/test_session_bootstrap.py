"""Tests for session bootstrap: what config a brand-new session starts from,
and whether a session created by another user is flagged.

The regression these guard: a fresh clone used to open on an empty config, and
a session inherited from another user (shared checkout, or a copied workspace
that carried .sessions/ along) silently kept their SLURM account and email.
"""

import shutil

import pytest
import yaml

import session as session_mod
from session import (
    blank_placeholders,
    create_session,
    default_seed_config,
    foreign_session_owner,
    list_sessions,
    load_registry,
    migrate_legacy,
)

REAL_REPO_ROOT = session_mod.REPO_ROOT
REAL_TEMPLATE = REAL_REPO_ROOT / "config.kempner.template.yaml"


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A throwaway repo root with the real Kempner template and no .sessions/.

    Points every module-level path in session.py at tmp_path so nothing touches
    the developer's own .sessions/ directory.
    """
    shutil.copy2(REAL_TEMPLATE, tmp_path / "config.kempner.template.yaml")
    monkeypatch.setattr(session_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path / ".sessions")
    monkeypatch.setattr(session_mod, "REGISTRY_FILE", tmp_path / ".sessions" / "registry.json")
    monkeypatch.setattr(
        session_mod, "KEMPNER_TEMPLATE", tmp_path / "config.kempner.template.yaml"
    )
    monkeypatch.setenv("USER", "newuser")
    return tmp_path


def _active_config(fake_repo) -> dict:
    registry = load_registry()
    cfg_path = fake_repo / ".sessions" / registry["active_session_id"] / "config.yaml"
    return yaml.safe_load(cfg_path.read_text()) or {}


class TestBlankPlaceholders:
    def test_replaces_angle_bracket_markers(self):
        out = blank_placeholders({"slurm": {"account": "<YOUR_KEMPNER_ACCOUNT>"}})
        assert out["slurm"]["account"] == ""

    def test_leaves_real_values_alone(self):
        cfg = {"slurm": {"account": "kempner_x_lab", "email": "a@b.edu"}}
        assert blank_placeholders(cfg) == cfg

    def test_recurses_into_lists_and_preserves_non_strings(self):
        cfg = {"a": ["<FILL_ME>", "keep"], "b": 10, "c": None, "d": True}
        assert blank_placeholders(cfg) == {"a": ["", "keep"], "b": 10, "c": None, "d": True}

    def test_does_not_eat_comparison_like_strings(self):
        """Only a whole string wrapped in <> is a placeholder."""
        cfg = {"note": "use <8 GPUs", "expr": "a < b > c"}
        assert blank_placeholders(cfg) == cfg


class TestFreshCloneSeed:
    """A clone with no config.yaml must still open on working cluster defaults."""

    def test_seeds_from_kempner_template(self, fake_repo, monkeypatch):
        monkeypatch.setenv("PROTFORGE_ROOT", "/n/lab/newuser")
        migrate_legacy()
        cfg = _active_config(fake_repo)

        # Not the old empty-dict bootstrap.
        assert cfg != {}
        # Shared cluster resources arrive filled in.
        assert cfg["msa"]["mmseq2_db"].startswith("/n/holylfs06/LABS/kempner_shared/")
        assert cfg["boltz"]["cache_dir"].startswith("/n/holylfs06/LABS/kempner_shared/")
        # The user's workspace is expanded from their environment.
        assert cfg["output"]["parent_dir"] == "/n/lab/newuser/outputs"

    def test_user_specific_placeholders_are_blank_not_someone_elses(self, fake_repo, monkeypatch):
        monkeypatch.setenv("PROTFORGE_ROOT", "/n/lab/newuser")
        migrate_legacy()
        slurm = _active_config(fake_repo)["slurm"]
        assert slurm["account"] == ""
        assert slurm["email"] == ""

    def test_unset_protforge_root_keeps_literal_placeholder(self, fake_repo, monkeypatch):
        """Seeding must not require PROTFORGE_ROOT — the field still shows what
        belongs there, and check_unexpanded_paths blocks the launch."""
        monkeypatch.delenv("PROTFORGE_ROOT", raising=False)
        migrate_legacy()
        cfg = _active_config(fake_repo)
        assert cfg["output"]["parent_dir"] == "${PROTFORGE_ROOT}/outputs"
        assert cfg["msa"]["mmseq2_db"].startswith("/n/holylfs06/LABS/kempner_shared/")

    def test_existing_repo_config_wins_over_template(self, fake_repo, monkeypatch):
        """A user who followed the documented `cp ... config.yaml` setup step
        keeps their own file."""
        monkeypatch.setenv("PROTFORGE_ROOT", "/n/lab/newuser")
        (fake_repo / "config.yaml").write_text(
            yaml.dump({"slurm": {"account": "kempner_mine_lab"}, "marker": "mine"})
        )
        migrate_legacy()
        cfg = _active_config(fake_repo)
        assert cfg["marker"] == "mine"
        assert cfg["slurm"]["account"] == "kempner_mine_lab"

    def test_unreadable_repo_config_falls_back_to_template(self, fake_repo, monkeypatch):
        monkeypatch.setenv("PROTFORGE_ROOT", "/n/lab/newuser")
        (fake_repo / "config.yaml").write_text("{{ not: valid: yaml")
        cfg = default_seed_config()
        assert cfg["output"]["parent_dir"] == "/n/lab/newuser/outputs"

    def test_no_template_and_no_config_yields_empty(self, fake_repo, monkeypatch):
        (fake_repo / "config.kempner.template.yaml").unlink()
        assert default_seed_config() == {}

    def test_migrate_is_a_noop_once_bootstrapped(self, fake_repo, monkeypatch):
        monkeypatch.setenv("PROTFORGE_ROOT", "/n/lab/newuser")
        migrate_legacy()
        first = load_registry()["active_session_id"]
        migrate_legacy()
        assert load_registry()["active_session_id"] == first
        assert len(list_sessions()) == 1


class TestCreateSession:
    def test_none_seeds_from_defaults(self, fake_repo, monkeypatch):
        monkeypatch.setenv("PROTFORGE_ROOT", "/n/lab/newuser")
        s = create_session("Run A")
        cfg = yaml.safe_load(s.config_path.read_text())
        assert cfg["output"]["parent_dir"] == "/n/lab/newuser/outputs"

    def test_explicit_empty_dict_stays_empty(self, fake_repo):
        """The sidebar's "Empty config" option must not get cluster defaults."""
        s = create_session("Blank", base_config={})
        assert yaml.safe_load(s.config_path.read_text()) == {}

    def test_explicit_config_is_written_verbatim(self, fake_repo):
        s = create_session("Clone", base_config={"marker": "cloned"})
        assert yaml.safe_load(s.config_path.read_text()) == {"marker": "cloned"}

    def test_records_creating_user(self, fake_repo):
        s = create_session("Run A", base_config={})
        entry = next(e for e in list_sessions() if e["id"] == s.id)
        assert entry["created_by"] == "newuser"


class TestForeignSessionOwner:
    def test_flags_a_session_created_by_someone_else(self, fake_repo, monkeypatch):
        s = create_session("Theirs", base_config={})
        monkeypatch.setenv("USER", "otheruser")
        assert foreign_session_owner(s.id) == "newuser"

    def test_own_session_is_not_flagged(self, fake_repo):
        s = create_session("Mine", base_config={})
        assert foreign_session_owner(s.id) is None

    def test_legacy_session_without_owner_is_not_flagged(self, fake_repo):
        """Registries written before created_by existed: unknown, not foreign."""
        s = create_session("Old", base_config={})
        registry = load_registry()
        for entry in registry["sessions"]:
            entry.pop("created_by", None)
        session_mod.save_registry(registry)
        assert foreign_session_owner(s.id) is None

    def test_unknown_session_id_is_not_flagged(self, fake_repo):
        create_session("Mine", base_config={})
        assert foreign_session_owner("deadbeef") is None
