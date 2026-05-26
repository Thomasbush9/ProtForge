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

Base image: `ubuntu:22.04`. The CUDA runtime + cuDNN come from the
`torch+cu124` wheel (which bundles them under `torch/lib/`); the NVIDIA
driver libs are injected at runtime by `singularity exec --nv` from the host.
No host `module load cuda/cudnn` is needed in container mode — those only
apply to the bash/conda fallback path documented in `workflow/rules/*.smk`.

Total SIF size: ~12 GB.

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
+ build cache. `containers/build.sh` does **not** default to `~/sifs` (non-interactive
`bash containers/build.sh` never reads `~/.bashrc`, so exports there are invisible unless
you `source ~/.bashrc` first).

**Output SIF path** is chosen in order: `-o` / `--output`, else `PROTFORGE_SIF_DIR/protforge-gpu.sif`,
else `PROTFORGE_ROOT/sifs/protforge-gpu.sif`. If none apply, the script exits with an error.

When `PROTFORGE_ROOT` is set and `SINGULARITY_CACHEDIR` / `SINGULARITY_TMPDIR` are unset,
`build.sh` exports them to `$PROTFORGE_ROOT/sing_cache` and `$PROTFORGE_ROOT/sing_tmp`
and creates those directories.

**IMPORTANT.** `PROTFORGE_ROOT` is a **workspace** directory that holds `sifs/`,
`sing_cache/`, `sing_tmp/` *alongside* the repo checkout — it must **not** be
the repo path itself. If you set `PROTFORGE_ROOT` to the repo, the build's
`sing_tmp/` lands inside the repo and `%files . /opt/protforge` recurses into
itself with `cp: cannot copy a directory into itself`. The repo lives as a
sibling, e.g. `$PROTFORGE_ROOT/ProtForge/`.

```bash
# Example for Sabatini lab on Kempner — PROTFORGE_ROOT is the *parent* of the
# repo, not the repo itself. Layout under it:
#   $PROTFORGE_ROOT/
#     ProtForge/         <- the repo (git checkout)
#     sifs/              <- output SIFs
#     sing_cache/        <- singularity layer cache
#     sing_tmp/          <- singularity build staging
export PROTFORGE_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/<you>
mkdir -p "$PROTFORGE_ROOT/sifs"

# Optional explicit overrides (same layout as above):
export PROTFORGE_SIF_DIR=$PROTFORGE_ROOT/sifs
export SINGULARITY_CACHEDIR=$PROTFORGE_ROOT/sing_cache
export SINGULARITY_TMPDIR=$PROTFORGE_ROOT/sing_tmp
mkdir -p "$PROTFORGE_SIF_DIR" "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

# To persist for interactive shells, append exports to ~/.bashrc; for one-shot
# builds and sbatch, export in the same script that invokes build.sh.
```

**HF_TOKEN is strongly recommended.** Kempner's compute/login nodes share
egress IPs across many users; HF Hub's per-IP anonymous rate limit gets hit
fast (typically on the ESM-C download, which fetches many small files via
the esm SDK). Anonymous builds work *sometimes* but fail unpredictably with
`LocalEntryNotFoundError` or `429`. Grab a read-only token at
<https://huggingface.co/settings/tokens> and export it before building:

```bash
export HF_TOKEN=hf_xxx  # in the same shell as build.sh, NOT in ~/.bashrc
```

`build.sh` stages the token to a mode-600 file under `SINGULARITY_TMPDIR`
and bind-mounts it at `/run/secrets/hf_token:ro` for the build; `%post`
reads it with shell-trace suppressed so it never lands in the build log.
The token is unset before the build context is copied to `/opt/protforge`
and removed from the host on script exit. The image's runtime
`%environment` does NOT set `HF_TOKEN`, so it never leaks at runtime.

Then one of:

```bash
# (a) Build locally from the def file (tries --fakeroot)
bash containers/build.sh                       # needs PROTFORGE_ROOT or PROTFORGE_SIF_DIR or -o

# (b) Pull a pre-built image from a registry (same output rules as (a); use -o if needed)
bash containers/build.sh --from-docker docker://ghcr.io/<owner>/protforge-gpu:latest

# (c) Custom output / dry run
bash containers/build.sh -o /path/to/out.sif
bash containers/build.sh --dry-run

# (d) Pass an HF token explicitly (overrides $HF_TOKEN env)
bash containers/build.sh --hf-token hf_xxx
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
bash containers/test/smoke.sh                  # image: PROTFORGE_SIF_DIR, else PROTFORGE_ROOT/sifs, else ~/sifs
bash containers/test/smoke.sh -i /path/to/sif
```

Validates: GPU visible, PyTorch+CUDA work, all tools importable, baked
weights load, ESMFold folds a short sequence end-to-end. Does not test
MSA or Boltz (those need the bind-mounted DBs).

## Runtime usage on Kempner

Manual invocation (same-path read-only DB binds — the template default):

```bash
singularity exec --nv \
    -B /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db:ro \
    -B /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/colabfold_db:ro \
    -B /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db:ro \
    -B "$PWD":"$PWD" \
    "$PROTFORGE_ROOT/sifs/protforge-gpu.sif" \
    boltz predict ...
```

Via Snakemake (already wired): set `containers.gpu` in your config to the SIF
path. The `containers.bind_paths` default in `config.template.yaml` already
mounts the three shared DBs at the same host paths inside the container,
read-only, so the existing `msa.mmseq2_db` / `boltz.cache_dir` values work
unchanged. The `container_cmd()` helper in `Snakefile` dispatches automatically.

If you prefer the canonical `/data/colabfold_db` / `/data/boltz_db` mounts
documented in the def file's `%environment`, change each `bind_paths` entry
to `host:/data/<name>:ro` AND update the matching stage config values
(`msa.mmseq2_db`, `msa.colabfold_db`, `boltz.cache_dir`) to the `/data/...`
side.

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
