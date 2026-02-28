# ProtForge Container Design

Full containerization reference for team discussion.

## Overview

Each pipeline stage gets its own Singularity/Apptainer `.sif` image. The Snakemake rules detect whether a container path is configured and dispatch accordingly:

- **Container mode**: `singularity exec --nv -B <bind_paths> <image.sif> <command>`
- **Legacy mode**: `module load` + `conda activate` (current behavior)

This dual-mode approach lets users migrate gradually — one stage at a time — without breaking existing setups.

## Container Images

| Stage | Image name | Base | Key packages |
|-------|-----------|------|-------------|
| MSA | `protforge-colabfold.sif` | nvidia/cuda:12.4.1-runtime-ubuntu22.04 | ColabFold, MMseqs2 |
| Boltz | `protforge-boltz.sif` | nvidia/cuda:12.4.1-runtime-ubuntu22.04 | Boltz, PyTorch 2.x |
| ESM | `protforge-esm.sif` | nvidia/cuda:12.4.1-runtime-ubuntu22.04 | fair-esm, PyTorch 2.x |
| ES | `protforge-pdanalysis.sif` | ubuntu:22.04 + OpenMPI | PDAnalysis, mpi4py |

## CUDA Version Rationale

Using CUDA 12.4.1 as the base for GPU stages because:
- Compatible with Kempner cluster's NVIDIA A100/H100 drivers (driver >= 525)
- Matches PyTorch 2.x prebuilt wheels (torch+cu124)
- Avoids the CUDA 12.6+ breaking changes in cuDNN APIs
- Singularity `--nv` injects host driver libs, so the container only needs the CUDA runtime

## Bind Mount Strategy

Default bind paths: `/n/holylfs06,/n/home06`

These cover:
- `/n/holylfs06` — shared lab storage (databases, model weights, shared envs)
- `/n/home06` — user home directories (configs, outputs, sif files)

Users can override via `containers.bind_paths` in `config.yaml`. Additional paths (e.g., scratch filesystems) can be comma-separated.

**Important**: Singularity automatically binds `/tmp`, `/proc`, `/sys`, `/dev`. The `--nv` flag handles GPU device files and driver libraries.

## Dockerfile Specifications

### ColabFold (MSA stage)

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    wget git build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Install MMseqs2
RUN wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz \
    && tar xzf mmseqs-linux-avx2.tar.gz \
    && mv mmseqs/bin/mmseqs /usr/local/bin/ \
    && rm -rf mmseqs mmseqs-linux-avx2.tar.gz

# Install ColabFold
RUN pip install colabfold[all]

# Verify
RUN colabfold_search --help
```

### Boltz (Structure prediction)

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install boltz torch --index-url https://download.pytorch.org/whl/cu124

# Verify
RUN boltz --help
```

### ESM (Embeddings)

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install fair-esm torch --index-url https://download.pytorch.org/whl/cu124

# Copy the run_esm.py script into the container
COPY slurm_scripts/run_esm.py /opt/protforge/run_esm.py

# Verify
RUN python3 -c "import esm; print('ESM OK')"
```

### PDAnalysis (ES stage)

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install mpi4py numpy scipy

# Clone and install PDAnalysis
RUN git clone https://github.com/<ORG>/PDAnalysis.git /opt/pdanalysis
WORKDIR /opt/pdanalysis
RUN pip install -e .

# Verify
RUN python3 -c "import PDAnalysis; print('PDAnalysis OK')"
```

## Build and Pull Workflow

### Building locally (for development)

```bash
# Build Docker image
docker build -t protforge-boltz:latest -f containers/Dockerfile.boltz .

# Convert to Singularity .sif
singularity build protforge-boltz.sif docker-daemon://protforge-boltz:latest
```

### Pulling from registry (for users)

```bash
# Pull pre-built images
singularity pull docker://ghcr.io/ORG/protforge-colabfold:latest
singularity pull docker://ghcr.io/ORG/protforge-boltz:latest
singularity pull docker://ghcr.io/ORG/protforge-esm:latest
singularity pull docker://ghcr.io/ORG/protforge-pdanalysis:latest

# Move to a known location
mkdir -p ~/sifs
mv protforge-*.sif ~/sifs/
```

### CI/CD

Proposed GitHub Actions workflow:
1. On push to `main` with changes in `containers/`, build all affected images
2. Push to `ghcr.io/ORG/protforge-<stage>:<tag>`
3. Tag with both `latest` and the git SHA

## Configuration

In `config.yaml`:

```yaml
containers:
  colabfold: /n/home06/user/sifs/protforge-colabfold.sif
  boltz: /n/home06/user/sifs/protforge-boltz.sif
  esm: /n/home06/user/sifs/protforge-esm.sif
  pdanalysis: /n/home06/user/sifs/protforge-pdanalysis.sif
  bind_paths: "/n/holylfs06,/n/home06"
```

Leave any `.sif` path empty to use legacy mode for that stage. This allows mixed mode — e.g., container for Boltz but legacy for MSA.

## How It Works in Snakemake

The `container_cmd()` helper in `Snakefile` reads the config and returns either:
- `"singularity exec --nv -B /n/holylfs06 -B /n/home06 /path/to.sif"` (container mode)
- `""` (legacy mode)

Each rule's shell block checks `if [ -n "{params.container_cmd}" ]` to dispatch.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| CUDA version mismatch | GPU ops fail silently or crash | Pin CUDA runtime in Dockerfile; test on target hardware before deploying |
| Missing bind mounts | FileNotFoundError at runtime | Validate paths in a pre-flight check rule; document required bind_paths |
| Image size (multi-GB) | Slow pulls, disk quota issues | Use multi-stage Docker builds; share base layers across stages |
| OpenMPI version mismatch (ES) | MPI init failures | Match container OpenMPI to host PMIx version; use `srun --mpi=pmix` |
| Stale images | Bug fixes not picked up | Tag images with version; add `--pull` option to build script |
| Permission issues | Singularity runs as user, not root | Avoid root-only paths in containers; test with non-root user |

## Migration Plan

1. **Phase 1** (current): Add dual-mode support to Snakemake rules. All `.sif` paths default to empty (legacy mode). No behavior change for existing users.
2. **Phase 2**: Build and test container images for each stage. Validate on Kempner cluster with real datasets.
3. **Phase 3**: Publish images to GHCR. Update docs with pull instructions. Encourage adoption.
4. **Phase 4**: Deprecate legacy mode once all users have migrated. Remove `module load` / `conda activate` fallbacks.

## Testing Strategy

```bash
# 1. Dry run with containers empty — should produce identical DAG
snakemake --profile profiles/slurm/ -n

# 2. Set one container path, dry run — verify singularity exec appears
# Edit config.yaml: containers.boltz: /path/to/protforge-boltz.sif
snakemake --profile profiles/slurm/ -n 2>&1 | grep "singularity exec"

# 3. Full run with one stage containerized
snakemake --profile profiles/slurm/ --until boltz_complete

# 4. Compare outputs between legacy and container runs
diff -r legacy_output/ container_output/
```
