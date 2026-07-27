---
name: setup
description: >-
  Walk a new user through a first-time ProtForge install on the Kempner cluster.
  Use when the user wants to set up ProtForge, do a first-time install, install
  the pipeline, configure a new user, or build the container. Interviews the
  researcher for their workspace/SLURM/storage choices, then guides and runs the
  real setup: workspace dirs, config.yaml, SIF build (on a compute node), model
  weight download, and the smoke test.
---

# Set up ProtForge (first-time install on Kempner)

Drive a brand-new user from a fresh repo clone to a validated, runnable
pipeline. This skill is the conversational driver over the existing setup
scripts — `containers/build.sh`, `scripts/download_models.py`, and the
per-stage `containers/test/*_test_image.sh` checks. The authoritative reference
is `docs/CLUSTER_SETUP.md`; defer to it and `containers/README.md` when in doubt.

Do **not** reimplement any setup logic. Run the scripts.

ProtForge runs **one container image per GPU stage** (msa / boltz / esm /
openfold), not a single mega-SIF. You only build the stages the user will run.
There is no `setup.sh` and no legacy conda path — the container path is the
only supported install. If `--fakeroot` is denied on the build partition, pull
prebuilt images with `--from-docker` instead of building.

## Shared vs. user — set expectations up front

**Already on Kempner, no setup needed** (read-only, point the config at them):

- ColabFold / MMseqs2 DBs — `msa.mmseq2_db`, `msa.colabfold_db`
- Boltz model checkpoint — `boltz.cache_dir`
  (`/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/...`)

**Also possibly already present**: the per-stage SIFs and the ESM-C/ESMFold
weight cache. They are read-only, so a lab typically keeps one shared copy —
ask before planning a build (see step 2). If one exists, the whole build and
download half of this skill is skipped.

**The user must do** (this skill): pick a workspace, create dirs, fill
`config.yaml`, smoke-test — plus building the SIFs and downloading weights only
when no shared copy is available.

## Interactive steps run by the user, not by Claude

`salloc`, `module load`, and the in-allocation build/smoke commands need an
interactive shell this skill can't drive. Hand the user the exact command and
ask them to run it themselves via the `! <command>` prefix in their prompt
(or in their own terminal), then paste the result back. Claude does the
file edits and dry, non-interactive checks.

## 1. Interview — pin down the install

Ask only what you can't read from an existing `config.yaml`:

- **Asset root `PROTFORGE_ASSETS`** — where the images and weights live:
  `$PROTFORGE_ASSETS/{sifs/, models/hf/, models/openfold/, sing_cache/,
  sing_tmp/}`. Read-only once built, so it can be **one shared copy per lab**.
  Ask whether the user can read an existing one before planning a build.
- **Workspace root `PROTFORGE_ROOT`** — the **parent** of the repo, NOT the repo
  itself: `$PROTFORGE_ROOT/{ProtForge/, data/, outputs/, job_logs/}`. Per-user,
  never shared. Set it equal to `PROTFORGE_ASSETS` when the user is building
  their own images. Both must live on `/n/holylfs06` (or similar lab storage) —
  home dirs are too small for the SIFs + caches.
- **SLURM** — `account` (e.g. `kempner_yourpi_lab`), `partition`
  (default `kempner_requeue`), and notification `email`.
- **Output + logs** — `output.parent_dir` and `slurm.log_dir`, both on
  `/n/holylfs06` (not home).
- **Input** — `input.fasta_dir` (a dir of `.fasta`/`.fa`), if known yet.
- **Container source** — build locally from the def file (default; needs
  `--fakeroot` on the compute partition) **or** pull a pre-built image with
  `--from-docker docker://...`.
- **HF token (optional)** — only if shared egress hits Hugging Face rate
  limits. A read-only token from <https://huggingface.co/settings/tokens>,
  saved to a file (e.g. `~/.config/protforge/hf_token`).

## 2. Create the workspace dirs

**Ask first whether a shared asset copy exists.** The images and weights are
~120 GB and entirely read-only once built, so a lab usually keeps one copy that
everyone points at. On Kempner the bsabatini lab's lives at
`/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge-assets`. If the user can read
one, `PROTFORGE_ASSETS` points at it and **steps 3 and 4 below are skipped
entirely** — no build, no download.

```bash
# Shared assets (nothing to build, nothing to install):
export PROTFORGE_ASSETS=/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge-assets
export PROTFORGE_ROOT=/n/holylfs06/LABS/<your_lab>/Everyone/<you>
mkdir -p "$PROTFORGE_ROOT"/{data/fastas,outputs,job_logs}
module load python && mamba activate "$PROTFORGE_ASSETS/envs/host"

# Building your own instead — both variables point at your workspace:
export PROTFORGE_ASSETS="$PROTFORGE_ROOT"
mkdir -p "$PROTFORGE_ASSETS"/{sifs,models/hf,sing_cache,sing_tmp}
```

The shared assets tree also carries the **host environment** at
`$PROTFORGE_ASSETS/envs/host` (Snakemake + SLURM executor + Streamlit). It must
be *activated* in whatever shell runs `snakemake` or the webapp — the local
chunking rules shell out to a bare `python`, so an unactivated shell dies on the
first rule with `python: command not found` before any job is submitted. `mamba`
is only a shell function after `module load python`; chain them in one command.

`build.sh` auto-sets `SINGULARITY_CACHEDIR`/`SINGULARITY_TMPDIR` (and the
`APPTAINER_*` equivalents) under `$PROTFORGE_ASSETS` when they're unset, so you
don't have to export them — but the dirs above must exist on big storage.

## 3. Create and fill config.yaml

Copy the template, then Claude edits the copy in place — do **not** make the
user hand-edit. On Kempner prefer the **Kempner template**: the shared DBs,
partition, container runtime and `${PROTFORGE_ASSETS}`-relative SIF/cache paths
are already correct, so most must-fill fields below are pre-filled and only
`slurm.account`, `slurm.email` and `input.fasta_dir` are genuinely left to set.

```bash
cp config.kempner.template.yaml config.yaml   # Kempner (preferred)
# cp config.template.yaml config.yaml         # other clusters / full reference
```

A user who only ever drives the webapp can skip this: its first launch seeds a
Default session from the same Kempner template (expanding both env vars
when exported, blanking `slurm.account` / `slurm.email`). Doing the `cp` anyway
is still worth it — a repo-root `config.yaml` takes precedence when the session
is seeded, and it is what the CLI path reads.

The Kempner template keeps `${PROTFORGE_ASSETS}` (images, weight caches) and
`${PROTFORGE_ROOT}` (inputs, outputs, logs) placeholders, which the workflow
expands from the environment at load time. That means the user **must export
both in the shell they run `snakemake` from** — remind them, and add them to
their shell rc. If either is unset the run aborts with a clear error naming the
config key. If the user would rather not depend on the env vars, resolve the
placeholders to literal absolute paths when you edit `config.yaml`.

Fill only these **must-fill** fields (Claude edits `config.yaml`):

- `slurm.account`, `slurm.partition`, `slurm.email`, `slurm.log_dir`
- `output.parent_dir`
- `input.fasta_dir`
- `esmc.cache_dir` and `esmfold.cache_dir` → the **HOST** path to the HF cache
  (the dir that contains `hub/models--biohub--ESMC-*`, `…--ESMFold2-*`). The
  Snakemake rules bind it read-only into the container themselves
  (`-B {cache_dir}:/models/hf:ro`), so the container always sees `/models/hf` —
  do **NOT** set `cache_dir` to `/models/hf`; that is the in-container mount
  target, not a config value. **Always ASK the user for their host HF model-cache
  dir** rather than assuming `$PROTFORGE_ASSETS/models/hf` (some installs keep it
  elsewhere on lab storage).
- `openfold.cache_dir` (if running OpenFold) → host dir holding the OpenFold
  checkpoints (`ckpt_root`, `of3-p2-*.pt`); bound to `/models/openfold` and
  auto-loaded. It can be the **same** cache dir as the HF cache.
- `containers.*` → set after the build (step 6). The images are per-stage:
  `msa.sif` → `containers.colabfold`, `boltz.sif` → `containers.boltz`,
  `esm.sif` → both `containers.esmc` and `containers.esmfold`, `openfold.sif`
  → `containers.openfold`. If the user already has these built, point each
  `containers.<stage>` at its SIF and skip the build (step 4) entirely.

**Leave alone** the shared read-only paths that already work as shipped:
`msa.mmseq2_db`, `msa.colabfold_db`, `boltz.cache_dir`.

Leave `slurm.resources.*` and per-stage chunk sizes unset — those are sized by
the `run-pipeline` skill's estimator, not here.

## 4. Build the stage SIFs (compute node — NOT a login node)

The build needs `--fakeroot`, which login nodes lack. The user runs this in an
interactive allocation. Build only the stages they enabled in `config.yaml`
(`all` builds every stage). Give them the exact commands to run themselves:

```bash
# 1) Grab an interactive allocation (user runs this)
salloc -p test --account=<your_account> -t 4:00:00 --mem 32G --ntasks-per-node 4

# 2) Inside the allocation:
export PROTFORGE_ROOT=/n/holylfs06/LABS/<your_lab>/Everyone/<you>
cd "$PROTFORGE_ROOT/ProtForge"
bash containers/build.sh all      # writes $PROTFORGE_ASSETS/sifs/{msa,boltz,esm,openfold}.sif
#   or a subset: bash containers/build.sh boltz esm
```

Output-dir resolution: `-o/--output` (single stage) wins, else
`$PROTFORGE_SIF_DIR`, else `$PROTFORGE_ASSETS/sifs`, else `$PROTFORGE_ROOT/sifs`. Each `<stage>.sif` gets a
`.sha256` sidecar next to it for provenance.

Variants to offer:

```bash
bash containers/build.sh all --dry-run                              # print commands, build nothing
bash containers/build.sh boltz -o /custom/path/boltz.sif            # explicit output (one stage)
bash containers/build.sh boltz --from-docker docker://ghcr.io/<owner>/protforge-boltz:latest  # pull instead of build
```

If `--fakeroot` is denied, fall back to `--from-docker` per stage (or build with
Docker elsewhere and push to GHCR, then pull). Success prints
`Done. Image at: ...` and the size per stage; a build log is saved under
`$PROTFORGE_ROOT/build-logs/`. Ask the user to paste it back on failure.

`--from-docker` pulls work from any compute node (no fakeroot needed) and are
the Kempner-handbook-canonical path.

## 5. Download model weights (node with internet)

ESM-C and ESMFold weights are **not** baked into the SIF — they're downloaded
to the host cache and bind-mounted at `/models/hf`. Run once:

```bash
cd "$PROTFORGE_ROOT/ProtForge"
python scripts/download_models.py --cache-dir "$PROTFORGE_ASSETS/models/hf"
```

The script writes an HF cache (`HF_HOME`) under that dir, prints the resolved
commit SHAs, and the total size. Useful flags:

- `--models esmc esmfold` (default `all`) — limit which weights to pull.
- `--token-file ~/.config/protforge/hf_token` — use an HF token on rate limits.
- `--esmc-revision <sha>` / `--esmfold-revision <sha>` — pin exact commits for
  reproducibility (defaults resolve `main` → current HEAD SHA and log it).

If `PROTFORGE_ROOT` is exported, `--cache-dir` defaults to
`$PROTFORGE_ASSETS/models/hf`, so it can be omitted.

## 6. Point config.yaml at the stage SIFs

Once the SIFs exist, Claude sets the per-stage container paths:

```yaml
containers:
  runtime: auto      # auto | singularity | apptainer (Kempner binary is `singularity`)
  colabfold: /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/msa.sif
  boltz:     /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/boltz.sif
  esmc:      /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/esm.sif   # esm.sif serves esmc + esmfold
  esmfold:   /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/esm.sif
  openfold:  /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/openfold.sif
  # gpu: ""          # optional shared fallback used only when a per-stage key is empty
```

Only set the keys for stages the user enabled. Confirm each path matches the
build output from step 4.

## 7. Smoke test (GPU node)

There is no single smoke script — validate each built image with its per-stage
test script in `containers/test/`. Each checks: GPU visible → PyTorch+CUDA →
tools importable → mounted weights load → a tiny fold/predict end-to-end. Needs
the model cache from step 5. The user runs these in a GPU allocation, after
editing the path variables at the top of each script:

```bash
# user runs:
salloc -p kempner_h100 --account=<your_account> --gres=gpu:1 -t 30 --mem=32G
cd "$PROTFORGE_ROOT/ProtForge"
bash containers/test/esmfold2_test_image.sh   # ESM image (esm.sif) — no DB binds
bash containers/test/test_batch_esmc.sh       # ESM-C batch
bash containers/test/boltz_test_image.sh      # Boltz image — needs shared DB binds
bash containers/test/msa_test_image.sh        # MSA image — needs shared DB binds
bash containers/test/openfold3_test_image.sh  # OpenFold image
```

Run only the ones for stages the user enabled. Pass criterion: each script's own
success line. On failure, ask the user to paste the failing block — see
`containers/TESTING.md` for the common causes (no GPU node, CUDA mismatch,
missing `--nv`, unbound `/models/hf`).

The ESM smoke test does **not** exercise MSA or Boltz (those need the big DB
binds — use their own test scripts). The fuller MSA→Boltz→ESM→ESMFold
end-to-end recipe is in `containers/TESTING.md` "Step 2".

## 8. Confirm ready to run

With a passing smoke test and a filled `config.yaml`, the install is done.
Hand off to the `run-pipeline` skill (estimate resources, dry-run, launch), or
do a quick dry-run sanity check yourself:

```bash
snakemake --profile profiles/slurm/ -n
```

## Notes

- **Skip the heavy steps when assets pre-exist.** If the user already has built
  SIFs and a populated HF/OpenFold cache (common for returning users), steps 4
  (build) and 5 (download) are unnecessary — just verify the SIF + cache paths,
  fill `config.yaml`, and smoke-test. Ask up front what already exists.
- **`mamba` is a shell function, not on `PATH` after `module load python`.** It
  is only defined within the activated shell, and this skill's Bash calls do
  **not** persist shell state between invocations. Chain everything in ONE
  command: `module load python && mamba activate snakemake && <cmd>`. Running
  `module load python` and `mamba activate` as separate Bash calls fails with
  `mamba: command not found`.
- Lab notes (decisions, calibration, cluster paths) live in the vault under
  `~/Documents/Vault/Notes/Lab/protforge/`, not the repo.
- `CLAUDE.md` mentions a `download_tools.sh` — it no longer exists. The real
  setup is `containers/build.sh` + `scripts/download_models.py`.
- There is no `setup.sh` and no conda fallback: the per-stage container path is
  the only supported install. If `--fakeroot` is denied, pull prebuilt images
  with `containers/build.sh <stage> --from-docker docker://...`.
