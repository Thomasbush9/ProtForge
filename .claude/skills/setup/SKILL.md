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

**The user must do** (this skill): pick a workspace, create dirs, fill
`config.yaml`, build the per-stage SIFs, download ESM-C/ESMFold weights, smoke-test.

## Interactive steps run by the user, not by Claude

`salloc`, `module load`, and the in-allocation build/smoke commands need an
interactive shell this skill can't drive. Hand the user the exact command and
ask them to run it themselves via the `! <command>` prefix in their prompt
(or in their own terminal), then paste the result back. Claude does the
file edits and dry, non-interactive checks.

## 1. Interview — pin down the install

Ask only what you can't read from an existing `config.yaml`:

- **Workspace root `PROTFORGE_ROOT`** — the **parent** of the repo, NOT the repo
  itself. Layout: `$PROTFORGE_ROOT/{ProtForge/, sifs/, models/hf/,
  models/openfold/, sing_cache/, sing_tmp/}`. Keeping `sing_tmp/` and the SIFs
  as siblings of the repo (not inside it) also keeps the build cache off the
  checkout. Must live on `/n/holylfs06` (or similar lab storage) — home dirs
  are too small for the SIFs + caches.
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

Have the user export `PROTFORGE_ROOT` and create the tree (the repo should
already be cloned as `$PROTFORGE_ROOT/ProtForge`):

```bash
export PROTFORGE_ROOT=/n/holylfs06/LABS/<your_lab>/Everyone/<you>
mkdir -p "$PROTFORGE_ROOT/sifs" "$PROTFORGE_ROOT/models/hf" \
         "$PROTFORGE_ROOT/sing_cache" "$PROTFORGE_ROOT/sing_tmp"
```

`build.sh` auto-sets `SINGULARITY_CACHEDIR`/`SINGULARITY_TMPDIR` (and the
`APPTAINER_*` equivalents) under `$PROTFORGE_ROOT` when they're unset, so you
don't have to export them — but the dirs above must exist on big storage.

## 3. Create and fill config.yaml

Copy the template, then Claude edits the copy in place — do **not** make the
user hand-edit:

```bash
cp config.template.yaml config.yaml
```

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
  dir** rather than assuming `$PROTFORGE_ROOT/models/hf` (some installs keep it
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
bash containers/build.sh all      # writes $PROTFORGE_ROOT/sifs/{msa,boltz,esm,openfold}.sif
#   or a subset: bash containers/build.sh boltz esm
```

Output-dir resolution: `-o/--output` (single stage) wins, else
`$PROTFORGE_SIF_DIR`, else `$PROTFORGE_ROOT/sifs`. Each `<stage>.sif` gets a
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
python scripts/download_models.py --cache-dir "$PROTFORGE_ROOT/models/hf"
```

The script writes an HF cache (`HF_HOME`) under that dir, prints the resolved
commit SHAs, and the total size. Useful flags:

- `--models esmc esmfold` (default `all`) — limit which weights to pull.
- `--token-file ~/.config/protforge/hf_token` — use an HF token on rate limits.
- `--esmc-revision <sha>` / `--esmfold-revision <sha>` — pin exact commits for
  reproducibility (defaults resolve `main` → current HEAD SHA and log it).

If `PROTFORGE_ROOT` is exported, `--cache-dir` defaults to
`$PROTFORGE_ROOT/models/hf`, so it can be omitted.

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
