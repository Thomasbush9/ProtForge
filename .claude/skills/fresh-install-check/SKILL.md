---
name: fresh-install-check
description: >-
  Verify ProtForge still works for a brand-new user — simulate clone, config,
  dry run and session bootstrap, then check the real cluster assets. Use after a
  refactor, before handing the repo to a labmate, before a release or merge to
  main, or when asked to check the pipeline is not broken for new users, to
  smoke-test the install, or to validate setup end to end.
---

# Fresh-install check

Answers one question: **would someone cloning this repo right now get a working
pipeline?**

The unit suite cannot answer it. Every other test runs pure-Python against
`tmp_path`, and the session tests monkeypatch `session.REPO_ROOT` away — so
external paths and session bootstrap, the two things that actually break during
refactors, are invisible to it. That is why "tests pass but the pipeline is
broken for a new user" keeps happening.

Run this after any change to: `Snakefile`, `workflow/rules/*.smk`, the config
templates, `workflow/scripts/snake_helpers.py`, `webapp/session.py`,
`webapp/pipeline_ops.py`, or the setup docs.

## Layer 1 — the simulation (no cluster, ~1 min)

`tests/test_fresh_install.py` exports the tracked tree at **working-tree
content** (so it validates the change about to be committed, not the last
commit), points `PROTFORGE_ROOT`/`PROTFORGE_ASSETS` at temp dirs, walks the
documented setup, and asserts the workflow plans a run — and refuses to plan a
broken one.

```bash
# pytest and snakemake usually live in different envs, hence the explicit path.
cd "$PROTFORGE_ROOT/ProtForge"
PROTFORGE_SNAKEMAKE="$PROTFORGE_ASSETS/envs/host/bin/snakemake" \
  python -m pytest tests/test_fresh_install.py -v
```

Run the whole suite too — this file is included in the default `testpaths`:

```bash
PROTFORGE_SNAKEMAKE="$PROTFORGE_ASSETS/envs/host/bin/snakemake" python -m pytest -q
```

If `PROTFORGE_SNAKEMAKE` is unset and `snakemake` is not on `PATH`, the dry-run
tests **skip rather than fail**. A run reporting skips has not checked the
important half — say so plainly instead of reporting it as a pass.

Interpreting failures:

| Failing class | What broke |
|---|---|
| `TestCloneIsClean` | Developer state got committed, or a template stopped being tracked |
| `TestTemplatesAreCopyable` | A shipped template no longer parses or lost a field setup edits |
| `TestDryRunPlansTheRun` | The Snakefile/rule files no longer load, or a stage stopped reaching its sentinel |
| `TestBrokenInstallsAreRefused` | A broken install now plans cleanly — the silent-failure mode this exists to prevent |
| `TestWebappBootstrapsOnARealClone` | Session seeding broke against the real `REPO_ROOT` |

## Layer 2 — the real install (needs the cluster)

Layer 1 uses zero-byte stand-in SIFs and temp dirs, so it proves the *wiring* is
sound, not that this user's assets exist. Check the real ones read-only:

```bash
cd "$PROTFORGE_ROOT/ProtForge"
# Shared Kempner paths — deliberately NOT validated by preflight, so a user in
# another lab gets a clean check and then N failing jobs. Verify by hand.
ls -d /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/{mmseq2_db,colabfold_db,boltz_db}

# This user's images and weights, as the config actually names them. Pass the
# config to check — repo-root config.yaml for the CLI path, or a session's
# .sessions/<id>/config.yaml for the webapp path. A user who only ever drives
# the webapp has no repo-root config.yaml at all.
python -c "
import sys, yaml; sys.path.insert(0, 'webapp'); sys.path.insert(0, 'workflow/scripts')
from snake_helpers import expand_config
import pipeline_ops as P
cfg = expand_config(yaml.safe_load(open(sys.argv[1])))
errs = P.check_launch_inputs(cfg)['errors']
print('\n'.join(errs) if errs else 'preflight clean')
" config.yaml

# The dry run now enforces the same thing, so this must agree with the above:
snakemake --profile profiles/slurm/ -n
```

A disagreement between those last two is itself the bug — they share
`snake_helpers.check_stage_assets` precisely so they cannot drift.

## Layer 3 — a real one-sequence run (optional, ~20-40 min)

Only when the change touched rule bodies, container invocation or binds. The
recipe is `containers/TESTING.md` "Step 2" (ubiquitin, all stages, its own
config). Do not reimplement it — follow that doc.

## Reporting

Lead with the verdict a new user cares about: **would a clone work right now,
yes or no.** Then failures with the file and what broke. Mention skips
explicitly. If everything passed, say which layers ran — "layer 1 green, layer 2
not run" is an honest and useful answer; "all green" when only layer 1 ran is
not.

## Notes

- The test needs no GPU, no SLURM and no container. It never submits: `-n` only.
- `PROTFORGE_SKIP_ASSET_CHECK=1` bypasses the Snakefile's asset check, for
  planning a DAG off-cluster. It is an escape hatch, not a fix — never suggest
  it to work around a genuinely incomplete install.
- Adding a stage means adding it to `ALL_STAGES` in the test, or the new stage
  is silently unchecked.
- Related skills: `setup` (drives a real first-time install), `run-pipeline`
  (sizes and launches a real run).
