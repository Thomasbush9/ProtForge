"""Simulates a brand-new user: git clone -> config -> dry run -> webapp session.

Every other test in this suite is pure-Python against tmp_path, and the session
tests monkeypatch `session.REPO_ROOT` away. That leaves the two things that
actually break when we refactor — external paths and session bootstrap —
structurally invisible: no test parses the Snakefile, invokes snakemake, or
exercises the real REPO_ROOT.

This module closes that gap. It exports the tracked tree into a temp dir — a
faithful clone, so untracked config.yaml and .sessions/ are absent exactly as a
new user sees, but at working-tree content, so it validates the change you are
about to commit rather than the last one. It then points PROTFORGE_ROOT and
PROTFORGE_ASSETS at temp dirs, walks the documented setup, and asserts the
workflow both plans a run and refuses to plan a broken one.

No cluster, no GPU, no container: the images are zero-byte stand-ins, since
every check here is existence-only. `snakemake -n` never submits.

Marked `slow` — the dry-run subprocesses dominate at roughly a minute total.
Skip with `-m "not slow"`; run alone with `-m slow`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

# 76 aa, folds in every stage's smoke test — small enough that a real run of
# this input is cheap if someone graduates the fixture to an actual launch.
UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)

ALL_STAGES = ("msa", "boltz", "esmc", "esmfold", "openfold")

pytestmark = pytest.mark.slow


# pytest usually runs from an env that is not the host env (the host env carries
# snakemake, not pytest), so accept an explicit path to the binary rather than
# skipping the most valuable tests whenever the two do not coincide.
SNAKEMAKE = os.environ.get("PROTFORGE_SNAKEMAKE") or shutil.which("snakemake")

requires_snakemake = pytest.mark.skipif(
    not SNAKEMAKE,
    reason=(
        "snakemake binary not found. Activate the host env, or point at it: "
        "PROTFORGE_SNAKEMAKE=$PROTFORGE_ASSETS/envs/host/bin/snakemake"
    ),
)


@pytest.fixture(scope="module")
def fresh_install(tmp_path_factory):
    """A new user's world: cloned repo, workspace dirs, assets, config, input.

    Module-scoped — building it is a couple of seconds, but each dry run costs
    ~10s and there is no reason to redo the setup for every one. Nothing here
    mutates the tree; the tests that need a different config write their own
    copy and pass it with --configfile.
    """
    root = tmp_path_factory.mktemp("fresh")
    workspace = root / "workspace"
    assets = root / "assets"
    clone = workspace / "ProtForge"
    clone.mkdir(parents=True)

    # Tracked files at their *working-tree* content — what a clone would get if
    # you committed right now. Deliberately not `git archive HEAD` (which would
    # test the last commit and quietly ignore the change you are validating),
    # and not a plain copy (which would drag in the developer's own config.yaml,
    # .sessions/ and .snakemake/, hiding precisely the bugs we are hunting).
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z"],
        capture_output=True, check=True,
    ).stdout
    packed = subprocess.run(
        # -C must precede -T: GNU tar treats them positionally.
        ["tar", "-c", "-f", "-", "-C", str(REPO), "--null", "-T", "-"],
        input=tracked, check=True, capture_output=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(clone)], input=packed, check=True)

    for sub in ("data/fastas", "outputs", "job_logs"):
        (workspace / sub).mkdir(parents=True)
    for sub in ("sifs", "models/hf", "models/openfold", "sing_cache", "sing_tmp"):
        (assets / sub).mkdir(parents=True)
    # Existence-only checks, so empty files stand in for ~30 GB of images.
    for stage in ("msa", "boltz", "esm", "openfold"):
        (assets / "sifs" / f"{stage}.sif").touch()

    (workspace / "data/fastas/ubiquitin.fasta").write_text(f">ubiquitin\n{UBIQUITIN}\n")

    return {"root": root, "workspace": workspace, "assets": assets, "clone": clone}


@pytest.fixture(scope="module")
def env(fresh_install):
    """Environment for a subprocess run from inside the fresh clone."""
    e = dict(os.environ)
    e["PROTFORGE_ROOT"] = str(fresh_install["workspace"])
    e["PROTFORGE_ASSETS"] = str(fresh_install["assets"])
    return e


@pytest.fixture(scope="module")
def configured(fresh_install, env):
    """The documented step 3: copy the Kempner template, fill in the two fields.

    Mirrors `cp config.kempner.template.yaml config.yaml` plus the account/email
    edit the setup skill performs, so a template that stops being copyable
    fails here.
    """
    clone = fresh_install["clone"]
    cfg_path = clone / "config.yaml"
    shutil.copy(clone / "config.kempner.template.yaml", cfg_path)

    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["slurm"]["account"] = "kempner_test_lab"
    cfg["slurm"]["email"] = "test@example.edu"
    cfg["pipeline"] = {stage: True for stage in ALL_STAGES}
    cfg["esmc"]["sae"]["enabled"] = True
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def _dry_run(clone: Path, env: dict, configfile: Path | None = None):
    cmd = [SNAKEMAKE, "--profile", "profiles/slurm/", "-n"]
    if configfile:
        cmd += ["--configfile", str(configfile)]
    return subprocess.run(
        cmd, cwd=clone, env=env, capture_output=True, text=True, timeout=600
    )


def _write_variant(tmp_path: Path, base: Path, **overrides) -> Path:
    """A copy of `base` with top-level blocks merged over, for negative cases."""
    cfg = yaml.safe_load(base.read_text())
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    out = tmp_path / "variant.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


class TestCloneIsClean:
    """A fresh clone must not carry developer state, or the rest proves nothing."""

    def test_no_config_yaml(self, fresh_install):
        assert not (fresh_install["clone"] / "config.yaml").exists()

    def test_no_sessions_dir(self, fresh_install):
        assert not (fresh_install["clone"] / ".sessions").exists()

    def test_no_snakemake_state(self, fresh_install):
        assert not (fresh_install["clone"] / ".snakemake").exists()

    def test_templates_are_tracked(self, fresh_install):
        clone = fresh_install["clone"]
        assert (clone / "config.kempner.template.yaml").exists()
        assert (clone / "config.template.yaml").exists()

    def test_profile_is_tracked(self, fresh_install):
        assert (fresh_install["clone"] / "profiles/slurm/config.yaml").exists()


class TestTemplatesAreCopyable:
    """Both shipped templates must parse and carry the fields setup edits.

    config.template.yaml had no test at all — it uses a different placeholder
    convention from the Kempner one and could rot silently.
    """

    @pytest.mark.parametrize(
        "template", ["config.kempner.template.yaml", "config.template.yaml"]
    )
    def test_parses_and_has_editable_identity(self, fresh_install, template):
        cfg = yaml.safe_load((fresh_install["clone"] / template).read_text())
        for key in ("pipeline", "input", "output", "slurm", "containers"):
            assert key in cfg, f"{template} is missing the {key!r} block"
        assert "account" in cfg["slurm"] and "email" in cfg["slurm"]

    @pytest.mark.parametrize(
        "template", ["config.kempner.template.yaml", "config.template.yaml"]
    )
    def test_ships_stages_off_that_need_extra_weights(self, fresh_install, template):
        """openfold needs weights download_models.py does not fetch."""
        cfg = yaml.safe_load((fresh_install["clone"] / template).read_text())
        assert cfg["pipeline"].get("openfold") is not True


@requires_snakemake
class TestDryRunPlansTheRun:
    def test_all_stages_plan_cleanly(self, fresh_install, env, configured):
        result = _dry_run(fresh_install["clone"], env)
        assert result.returncode == 0, result.stdout + result.stderr
        # Every enabled stage must reach its terminal sentinel rule; a rule file
        # that silently stops being included would still exit 0 otherwise.
        for stage in ALL_STAGES:
            assert f"{stage}_complete" in result.stdout, (
                f"{stage} enabled but no {stage}_complete rule was planned"
            )
        assert "esmc_sae_complete" in result.stdout

    def test_outputs_land_under_the_configured_workspace(
        self, fresh_install, env, configured
    ):
        """Catches a rule that hardcodes a path instead of reading output.parent_dir."""
        result = _dry_run(fresh_install["clone"], env)
        assert str(fresh_install["workspace"] / "outputs") in result.stdout


@requires_snakemake
class TestBrokenInstallsAreRefused:
    """The failure mode this suite exists for: green dry run, doomed run.

    A missing image or weight cache used to plan a clean DAG and exit 0, so the
    documented `snakemake -n` sanity check green-lit an install where every job
    would die.
    """

    def test_missing_env_var_names_the_config_key(self, fresh_install, env, configured):
        broken = dict(env)
        del broken["PROTFORGE_ASSETS"]
        result = _dry_run(fresh_install["clone"], broken)
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "PROTFORGE_ASSETS" in combined
        # Naming the key is the point — "no such file" mid-run is the old failure.
        assert "cache_dir" in combined or "containers." in combined

    def test_absent_assets_are_refused(self, fresh_install, env, configured, tmp_path):
        """Nothing built yet: no SIFs, no weights."""
        broken = dict(env)
        broken["PROTFORGE_ASSETS"] = str(tmp_path / "never_built")
        result = _dry_run(fresh_install["clone"], broken)
        assert result.returncode != 0, (
            "dry run passed with no container images — "
            "every submitted job would have died"
        )
        assert "not found" in (result.stdout + result.stderr)

    def test_template_identity_placeholder_is_refused(
        self, fresh_install, env, configured, tmp_path
    ):
        """An unedited <YOUR_EMAIL> must never reach sbatch --mail-user."""
        variant = _write_variant(
            tmp_path, configured,
            slurm={"account": "kempner_test_lab", "email": "<YOUR_EMAIL>"},
        )
        result = _dry_run(fresh_install["clone"], env, configfile=variant)
        assert result.returncode != 0
        assert "slurm.email" in (result.stdout + result.stderr)

    def test_asset_check_is_escapable(self, fresh_install, env, configured, tmp_path):
        """Planning a DAG off-cluster stays possible via the documented escape."""
        loose = dict(env)
        loose["PROTFORGE_ASSETS"] = str(tmp_path / "never_built")
        loose["PROTFORGE_SKIP_ASSET_CHECK"] = "1"
        result = _dry_run(fresh_install["clone"], loose)
        assert result.returncode == 0, result.stdout + result.stderr


class TestWebappBootstrapsOnARealClone:
    """session.py resolves REPO_ROOT from __file__ at import.

    test_session_bootstrap.py monkeypatches that constant to tmp_path, so the
    real one is never exercised. Here it runs unpatched in the clone — a
    subprocess, because the module-level constants bind at import time and the
    repo under test is not the repo running pytest.
    """

    @staticmethod
    def _bootstrap(clone: Path, env: dict) -> subprocess.CompletedProcess:
        script = textwrap.dedent(
            """
            import json, pathlib, sys, yaml
            repo = pathlib.Path.cwd()
            sys.path.insert(0, str(repo / "webapp"))
            sys.path.insert(0, str(repo / "workflow" / "scripts"))
            import session as S, pipeline_ops as P

            assert S.REPO_ROOT == repo, f"REPO_ROOT={S.REPO_ROOT} cwd={repo}"
            S.migrate_legacy()
            reg = S.load_registry()
            entry = reg["sessions"][0]
            cfg = yaml.safe_load(S.Session(entry["id"]).config_path.read_text())
            print(json.dumps({
                "n_sessions": len(reg["sessions"]),
                "seeded_from": entry.get("seeded_from"),
                "created_by": entry.get("created_by"),
                "account": cfg.get("slurm", {}).get("account"),
                "fasta_dir": cfg.get("input", {}).get("fasta_dir"),
                "esmc_cache": cfg.get("esmc", {}).get("cache_dir"),
                "errors": P.check_launch_inputs(cfg)["errors"],
            }))
            """
        )
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=clone, env=env, capture_output=True, text=True, timeout=300,
        )

    @pytest.fixture(scope="class")
    @classmethod
    def bootstrapped(cls, fresh_install, env, configured):
        import json
        clone = fresh_install["clone"]
        shutil.rmtree(clone / ".sessions", ignore_errors=True)
        result = cls._bootstrap(clone, env)
        assert result.returncode == 0, result.stdout + result.stderr
        data = json.loads(result.stdout.strip().splitlines()[-1])
        yield data
        shutil.rmtree(clone / ".sessions", ignore_errors=True)

    def test_creates_exactly_one_default_session(self, bootstrapped):
        assert bootstrapped["n_sessions"] == 1

    def test_seeds_from_the_repo_config(self, bootstrapped, configured):
        assert bootstrapped["seeded_from"] == str(configured)

    def test_records_who_created_it(self, bootstrapped):
        """Absent created_by is what makes a session read as foreign later."""
        assert bootstrapped["created_by"]
        assert bootstrapped["created_by"] != "unknown"

    def test_env_vars_are_expanded_into_the_session(self, bootstrapped, fresh_install):
        assert bootstrapped["fasta_dir"] == str(fresh_install["workspace"] / "data/fastas")
        assert bootstrapped["esmc_cache"] == str(fresh_install["assets"] / "models/hf")
        assert "${" not in bootstrapped["esmc_cache"]

    def test_preflight_agrees_the_install_is_launchable(self, bootstrapped):
        """The UI and the dry run must reach the same verdict on one config."""
        assert bootstrapped["errors"] == []
