# ProtForge containers

ProtForge runs each GPU stage inside **its own Singularity/Apptainer image**
(not one mega-SIF). Large databases and model weights are **not** baked in —
they are bind-mounted from cluster storage at runtime.

| Stage | Def file | Output SIF | Serves | config key(s) |
|---|---|---|---|---|
| MSA | `msa.def` | `msa.sif` | `colabfold_search` + `mmseqs2` | `containers.colabfold` |
| Boltz | `boltz.def` | `boltz.sif` | Boltz structure prediction | `containers.boltz` |
| ESM | `esmfold_cu.def` | `esm.sif` | ESM-C embeddings + ESMFold2 | `containers.esmc`, `containers.esmfold` |
| OpenFold | `openfold.def` | `openfold.sif` | OpenFold3 | `containers.openfold` |

**Out of scope:** ES/PDAnalysis (separate MPI image, not yet built).

Why per-stage and not one image: the tools' dependency pins conflict
(colabfold vs numba vs boltz all want different numpy/cuda wheels). One image
per stage keeps each dependency set independent instead of solving an
impossible cross-tool constraint. Only build the stages you actually run.

## What's inside each image

Baked into the SIF: the stage's tools + PyTorch/CUDA userspace (from the
`torch+cuXXX` wheel, which bundles the CUDA runtime + cuDNN under `torch/lib/`).
The NVIDIA driver libs are injected at runtime by `singularity exec --nv` from
the host — no host `module load cuda/cudnn` is needed in container mode.

NOT baked (bind-mount at runtime):
- ColabFold MSA databases (~700 GB) — shared on Kempner.
- Boltz model checkpoint (~5 GB) — shared on Kempner.
- ESM-C / ESMFold HF cache and OpenFold weights → mount to `/models/hf`,
  `/models/openfold`. Populated by `scripts/download_models.py`.

## Build / fetch on Kempner

Two modes, both via `containers/build.sh`:

1. **`--from-def` (default):** local `singularity build --fakeroot` from the
   stage def. Works if `--fakeroot` is permitted on your partition.
2. **`--from-docker docker://…`:** pull a prebuilt image (one URL per stage).
   Handbook-canonical; works from any compute node.

**Always run builds/pulls from an interactive allocation, NOT a login node**
([Kempner handbook](https://handbook.eng.kempnerinstitute.harvard.edu/s1_high_performance_computing/development_and_runtime_envs/containerization.html)):

```bash
salloc --partition=test --account=<your_account> \
       --nodes=1 --ntasks-per-node=4 --mem-per-cpu=3200M --time=4:00:00
```

**Pick an install location with space.** Home-dir quota is too small for the
SIFs + build cache. `build.sh` writes each `<stage>.sif` into an output dir
resolved in this order: `-o/--output` (single stage only), else
`$PROTFORGE_SIF_DIR`, else `$PROTFORGE_ROOT/sifs`. If none are set it errors.
When `PROTFORGE_ROOT` is set and `SINGULARITY_CACHEDIR`/`SINGULARITY_TMPDIR`
are unset, `build.sh` points them at `$PROTFORGE_ROOT/sing_cache` and
`$PROTFORGE_ROOT/sing_tmp`.

> Non-interactive `bash containers/build.sh` never reads `~/.bashrc`, so
> exports there are invisible unless you `source ~/.bashrc` first or export in
> the same script/allocation.

```bash
# Workspace layout — PROTFORGE_ROOT holds sifs/, caches, and the repo as siblings:
#   $PROTFORGE_ROOT/
#     ProtForge/         <- the repo (git checkout)
#     sifs/              <- output SIFs (msa.sif, boltz.sif, esm.sif, openfold.sif)
#     sing_cache/        <- singularity layer cache
#     sing_tmp/          <- singularity build staging
#     models/hf/         <- ESM-C + ESMFold HF cache
#     models/openfold/   <- OpenFold3 weights + CCD cache
export PROTFORGE_ROOT=/n/holylfs06/LABS/<your_lab>/Everyone/<you>
mkdir -p "$PROTFORGE_ROOT/sifs" "$PROTFORGE_ROOT/models/hf" "$PROTFORGE_ROOT/models/openfold"
```

Then one of:

```bash
# (a) Build every stage image from its def (tries --fakeroot)
bash containers/build.sh all

# (b) Build a subset
bash containers/build.sh boltz esm

# (c) Pull one prebuilt image instead of building
bash containers/build.sh boltz --from-docker docker://ghcr.io/<owner>/protforge-boltz:latest

# (d) Custom output path (single stage) / dry run
bash containers/build.sh boltz -o /path/to/boltz.sif
bash containers/build.sh all --dry-run
```

`build.sh` writes a `<stage>.sif.sha256` sidecar next to each image; the per-run
provenance manifest reads it instead of re-hashing multi-GB SIFs on every launch.

If `--fakeroot` fails ("fakeroot not allowed"), build the image with Docker +
push to GHCR elsewhere, then `--from-docker` it on Kempner.

## Download model weights

Run once on a node with internet access (MSA and Boltz need no downloaded
weights — their DBs/checkpoints are shared on Kempner):

```bash
cd "$PROTFORGE_ROOT/ProtForge"
python -m venv "$PROTFORGE_ROOT/host-env"
source "$PROTFORGE_ROOT/host-env/bin/activate"
pip install -r requirements-host.txt

python scripts/download_models.py --cache-dir "$PROTFORGE_ROOT/models/hf"
```

If shared cluster egress hits Hugging Face rate limits, use a read-only token
from <https://huggingface.co/settings/tokens>:

```bash
python scripts/download_models.py \
    --cache-dir "$PROTFORGE_ROOT/models/hf" \
    --token-file ~/.config/protforge/hf_token
```

## Smoke test

There is no single `smoke.sh`. Validate each built image with its per-stage
test script (each takes the SIF path and a small fixture from `test/`):

```bash
# On a GPU node: salloc -p kempner_h100 --gres=gpu:1 -t 30 --mem=32G
bash containers/test/esmfold2_test_image.sh   # ESM image
bash containers/test/boltz_test_image.sh      # Boltz image (needs the shared DB binds)
bash containers/test/msa_test_image.sh        # MSA image (needs the shared DB binds)
bash containers/test/openfold3_test_image.sh  # OpenFold image
```

Edit the path variables at the top of each script (input fixture, output dir,
SIF path) before running. `containers/TESTING.md` has the full end-to-end recipe.

## Point config.yaml at the built SIFs

```yaml
containers:
  runtime: auto                    # auto | singularity | apptainer
  colabfold: /…/sifs/msa.sif
  boltz:     /…/sifs/boltz.sif
  esmc:      /…/sifs/esm.sif       # esmc + esmfold share the ESM image
  esmfold:   /…/sifs/esm.sif
  openfold:  /…/sifs/openfold.sif
  # gpu: ""                        # optional shared fallback if a stage key is empty
```

Each rule binds its own inputs, shared DBs, and model cache read-only (e.g. the
ESM rules map the host HF cache to `/models/hf`, so `esmc.cache_dir: /models/hf`
works offline). The `container_cmd()` helper in `Snakefile` dispatches into the
right image per stage automatically. See `config.template.yaml` for the full
annotated block and `docs/CLUSTER_SETUP.md` for the end-to-end walkthrough.

## Env vars (in-container defaults)

- `HF_HOME` (`/models/hf`): host-mounted Hugging Face cache populated by
  `scripts/download_models.py`.
- `TORCH_HOME` (`/models/torch`): torch hub cache mount point.
- `OPENFOLD_CACHE` (`/models/openfold`): OpenFold weights + CCD cache mount.

## Not yet covered (deferred)

- Local (non-SLURM) Snakemake profile — `profiles/local/`.
- ES/PDAnalysis MPI image.
- GHCR-published images (CI build). For now everything builds on Kempner.
