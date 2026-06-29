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
scripts — `containers/build.sh`, `scripts/download_models.py`, and
`containers/test/smoke.sh`. The authoritative reference is
`docs/CLUSTER_SETUP.md`; defer to it and `containers/README.md` when in doubt.

Do **not** reimplement any setup logic. Run the scripts.

This skill targets **Path A (container / single SIF)** — the reproducible path.
Path B (legacy conda via `setup.sh`) exists as a fallback; only steer there if
the container build can't work (e.g. `--fakeroot` is denied and there's no
pre-built image to pull). See `docs/CLUSTER_SETUP.md` "Path B".

## Shared vs. user — set expectations up front

**Already on Kempner, no setup needed** (read-only, point the config at them):

- ColabFold / MMseqs2 DBs — `msa.mmseq2_db`, `msa.colabfold_db`
- Boltz model checkpoint — `boltz.cache_dir`
  (`/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/...`)

**The user must do** (this skill): pick a workspace, create dirs, fill
`config.yaml`, build the SIF, download ESM-C/ESMFold weights, smoke-test.

## Interactive steps run by the user, not by Claude

`salloc`, `module load`, and the in-allocation build/smoke commands need an
interactive shell this skill can't drive. Hand the user the exact command and
ask them to run it themselves via the `! <command>` prefix in their prompt
(or in their own terminal), then paste the result back. Claude does the
file edits and dry, non-interactive checks.

## 1. Interview — pin down the install

Ask only what you can't read from an existing `config.yaml`:

- **Workspace root `PROTFORGE_ROOT`** — the **parent** of the repo, NOT the repo
  itself. Layout: `$PROTFORGE_ROOT/{ProtForge/, sifs/, models/hf/, sing_cache/,
  sing_tmp/}`. **Gotcha:** if `PROTFORGE_ROOT` is the repo path, the build's
  `sing_tmp/` lands inside the repo and `%files . /opt/protforge` dies with
  `cp: cannot copy a directory into itself`. Confirm the value is one level
  above the checkout. Must live on `/n/holylfs06` (or similar lab storage) —
  home dirs are too small for the ~15 GB SIF + caches.
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
  dir** rather than assuming `$PROTFORGE_ROOT/models/hf` (a prior install kept it
  next to the SIFs, e.g. `…/singularity_dev/images/cache_models`).
- `openfold.cache_dir` (if running OpenFold) → host dir holding the OpenFold
  checkpoints (`ckpt_root`, `of3-p2-*.pt`); bound to `/models/openfold` and
  auto-loaded. It can be the **same** `cache_models` dir as the HF cache.
- `containers.*` → set after the build (step 6). **Per-stage SIFs are a valid
  layout**: if the user already has separate `msa.sif`/`boltz.sif`/
  `fast_esmfold.sif` (serves both esmc + esmfold)/`openfold3.sif`, point each
  `containers.<stage>` at its own SIF and skip the build (step 4) entirely.

**Leave alone** the shared read-only paths that already work as shipped:
`msa.mmseq2_db`, `msa.colabfold_db`, `boltz.cache_dir`.

Leave `slurm.resources.*` and per-stage chunk sizes unset — those are sized by
the `run-pipeline` skill's estimator, not here.

## 4. Build the SIF (compute node — NOT a login node)

The build needs `--fakeroot`, which login nodes lack. The user runs this in an
interactive allocation. Give them the exact commands to run themselves:

```bash
# 1) Grab an interactive allocation (user runs this; ~30 min build)
salloc -p test --account=<your_account> -t 4:00:00 --mem 32G --ntasks-per-node 4

# 2) Inside the allocation:
export PROTFORGE_ROOT=/n/holylfs06/LABS/<your_lab>/Everyone/<you>
cd "$PROTFORGE_ROOT/ProtForge"
bash containers/build.sh          # writes $PROTFORGE_ROOT/sifs/protforge-gpu.sif
```

Output-path resolution: `-o/--output` wins, else `PROTFORGE_SIF_DIR/...`, else
`PROTFORGE_ROOT/sifs/protforge-gpu.sif`. `build.sh` refuses to build if the
output or tmp dir resolves inside the repo (the self-copy guard above).

Variants to offer:

```bash
bash containers/build.sh --dry-run                                  # print the command, build nothing
bash containers/build.sh -o /custom/path/protforge-gpu.sif          # explicit output
bash containers/build.sh --from-docker docker://ghcr.io/<owner>/protforge-gpu:latest  # pull instead of build
```

If `--fakeroot` is denied, fall back to `--from-docker` (or build with Docker
elsewhere and push to GHCR, then pull). Success prints
`Done. Image at: ...` and the size; a build log is saved under
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

## 6. Point config.yaml at the SIF

Once the SIF exists, Claude sets the container paths. Set every GPU-stage field
to the same SIF (single-image design):

```yaml
containers:
  runtime: auto      # auto | singularity | apptainer (Kempner binary is `singularity`)
  gpu: /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/protforge-gpu.sif
  colabfold: ""      # leave "" to fall back to containers.gpu, or set to the same SIF
  boltz: ""
  esmc: ""
  esmfold: ""
  openfold: ""
```

`containers.gpu` is the shared fallback used when a per-stage key is empty, so
setting just `gpu` covers all stages. Confirm the SIF path matches the build
output from step 4.

## 7. Smoke test (GPU node)

Validates: GPU visible → PyTorch+CUDA → tools importable → mounted weights load
→ ESMFold folds a short peptide end-to-end. Needs the model cache from step 5.
The user runs this in a GPU allocation:

```bash
# user runs:
salloc -p kempner_h100 --account=<your_account> --gres=gpu:1 -t 30 --mem=32G
cd "$PROTFORGE_ROOT/ProtForge"
bash containers/test/smoke.sh                 # or: -i /path/to/protforge-gpu.sif
```

NOTE: `containers/test/smoke.sh` is the entry point named by the docs, but it
may not be present in every checkout — verify first with `ls containers/test/`.
If it's missing, fall back to the per-stage scripts that ARE there
(`containers/test/*_test_image.sh`, e.g. `esmfold2_test_image.sh`,
`boltz_test_image.sh`) plus `containers/test/test_batch_esmc.sh`. Run the one
matching the stage you're about to use.

Pass criterion: `=== ALL SMOKE TESTS PASSED ===` (or each per-stage script's own
success line). Steps print `=== [N/7] ... ===` headers so failures localize. On
failure, ask the user to paste the failing block — see `containers/TESTING.md`
for the common causes per step (no GPU node, CUDA mismatch, missing `--nv`,
unbound `/models/hf`).

The smoke test does **not** exercise MSA or Boltz (those need the big DB binds).
The fuller MSA→Boltz→ESM→ESMFold end-to-end recipe is in
`containers/TESTING.md` "Step 2".

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
- Container path A is in late beta per `docs/CLUSTER_SETUP.md`; if it breaks,
  the legacy conda Path B (`bash setup.sh`) is the documented fallback.
