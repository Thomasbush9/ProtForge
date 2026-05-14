# ProtForge container

Single Singularity image bundling all GPU-stage tools and default model weights.

**Scope (this image):** MSA (colabfold_search + mmseqs2), Boltz, ESM-C, ESMFold.
**Out of scope:** ES/PDAnalysis (separate MPI image, not yet built).

## What's inside

Baked into the SIF:
- Tools: `boltz`, `colabfold_search`, `mmseqs`, `esm` SDK, HF `transformers`.
- Weights: `esmc_600m` (~2.5 GB), `facebook/esmfold_v1` (~4 GB), under `/opt/weights/hf`.

NOT baked (bind-mount at runtime):
- ColabFold MSA databases (~700 GB) → mount to `/data/colabfold_db`.
- Boltz model checkpoint (~5 GB) → mount to `/data/boltz_db`.

Total SIF size: ~15 GB.

## Build / fetch on Kempner

The Kempner handbook ([containerization page](https://handbook.eng.kempnerinstitute.harvard.edu/s1_high_performance_computing/development_and_runtime_envs/containerization.html))
documents `singularity pull docker://...` from an interactive compute node as
the canonical path. Local `singularity build` is not addressed by the handbook
but works if `--fakeroot` is permitted on your compute partition.

**Always run builds/pulls from an interactive allocation, NOT a login node.**

```bash
# Get an interactive shell first (handbook-recommended example):
salloc --partition=test --account=<your_account> \
       --nodes=1 --ntasks-per-node=4 --mem-per-cpu=3200M --time=4:00:00
```

**Pick install location.** Home dir quota is usually too small for ~15 GB SIF
+ ~7 GB build cache. Set these env vars to a shared lab dir before building:

```bash
# Example for Sabatini lab on Kempner:
export PROTFORGE_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/<you>/ProtForge
export PROTFORGE_SIF_DIR=$PROTFORGE_ROOT/sifs
export SINGULARITY_CACHEDIR=$PROTFORGE_ROOT/sing_cache
export SINGULARITY_TMPDIR=$PROTFORGE_ROOT/sing_tmp
mkdir -p "$PROTFORGE_SIF_DIR" "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

# Persist by appending the same lines to ~/.bashrc.
```

Then one of:

```bash
# (a) Build locally from the def file (tries --fakeroot)
bash containers/build.sh                       # writes $PROTFORGE_SIF_DIR/protforge-gpu.sif

# (b) Pull a pre-built image from a registry (e.g. GHCR after we wire CI)
bash containers/build.sh --from-docker docker://ghcr.io/<owner>/protforge-gpu:latest

# (c) Custom output / dry run
bash containers/build.sh -o /path/to/out.sif
bash containers/build.sh --dry-run
```

First build/pull downloads ~10 GB of packages + weights. `singularity build`
doesn't cache `%post` layers, so re-runs redownload everything; see `build.sh`
for the sandbox-iteration workflow if you're iterating on the def file.

If `--fakeroot` fails ("fakeroot not allowed" or similar), the fall-back is
to build the image somewhere with Docker + push to GHCR, then
`--from-docker` it on Kempner.

## Smoke test

After a build, on a GPU node (`salloc -p kempner_h100 --gres=gpu:1 -t 30 --mem=32G`):

```bash
bash containers/test/smoke.sh                  # uses ~/sifs/protforge-gpu.sif
bash containers/test/smoke.sh -i /path/to/sif  # custom path
```

Validates: GPU visible, PyTorch+CUDA work, all tools importable, baked
weights load, ESMFold folds a short sequence end-to-end. Does not test
MSA or Boltz (those need the bind-mounted DBs).

## Runtime usage on Kempner

Manual invocation:

```bash
singularity exec --nv \
    -B /n/holylfs06/LABS/kempner_shared/Everyone/workflow/colabfold/databases:/data/colabfold_db \
    -B /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db:/data/boltz_db \
    -B "$PWD":"$PWD" \
    ~/sifs/protforge-gpu.sif \
    boltz predict ...
```

Via Snakemake (already wired): set `containers.gpu` in your config to the SIF
path and `containers.bind_paths` to include the two `/data/*` mounts. The
existing `container_cmd()` helper in `Snakefile` dispatches automatically.

## Env vars

- `PROTFORGE_HOME` (default `/data/protforge`): host bind-mount convention for
  per-user cache (esm/esmfold overrides, output staging). Override with
  `--env PROTFORGE_HOME=/your/path` if mounted elsewhere.
- `HF_HOME` (default `/opt/weights/hf`): points at baked weights. Override to
  use a host-side HF cache (e.g. for trying newer model versions).
- `TORCH_HOME` (default `/opt/weights/torch`): same idea for torch hub.

## Not yet covered (deferred)

- Local (non-SLURM) Snakemake profile — `profiles/local/` will land after the
  SLURM-side image is validated.
- ES/PDAnalysis MPI image.
- GHCR-published image (CI build). For now everything builds on Kempner.
