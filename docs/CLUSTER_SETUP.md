# Kempner Cluster Setup Guide

This guide documents the shared resources and user-specific configuration needed to run ProtForge on the Kempner cluster.

## Shared Resources (Already Available)

These paths are on the shared filesystem and can be used by all Kempner users:

### MSA Generation (ColabFold/MMseqs2)
| Config Key | Shared Path | Description |
|------------|-------------|-------------|
| `msa.mmseq2_db` | `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db` | MMseqs2 sequence database |
| `msa.colabfold_db` | `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/colabfold_db` | ColabFold database |
| `msa.colabfold_bin` | `/n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz/localcolabfold/colabfold-conda/bin` | ColabFold binaries |

### Boltz Structure Prediction
| Config Key | Shared Path | Description |
|------------|-------------|-------------|
| `boltz.cache_dir` | `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db` | Boltz model weights |
| `boltz.colabfold_db` | `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/colabfold_db` | ColabFold database |
| `boltz.env_path` | `/n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz` | Boltz conda environment |

## User-Specific Setup Required

### 1. ESM Embeddings

**Current Status:** User-specific (needs shared setup)

ESM requires:
- A conda environment with `esm` package installed
- Model cache directory (downloads ~2.4GB on first run)

**Option A: Use existing shared Boltz env (if compatible)**
```yaml
esm:
  env_path: /n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz
  cache_dir: /n/holylfs06/LABS/kempner_shared/Everyone/workflow/esm_cache  # TO BE CREATED
```

**Option B: Create your own ESM environment**
```bash
# Create conda env
conda create -n esm python=3.10 -y
conda activate esm
pip install esm  # or: pip install fair-esm

# Pre-download model (optional but recommended)
python -c "from esm.models.esmc import ESMC; ESMC.from_pretrained('esmc_600m')"
```

Then set in config.yaml:
```yaml
esm:
  env_path: /n/home06/<YOUR_USER>/envs/esm
  cache_dir: /n/home06/<YOUR_USER>/cache/esm
  work_dir: /path/to/ProtForge  # where run_esm.py is located
```

**Model Cache:** The ESM model (`esmc_600m`) is downloaded from HuggingFace on first run. Set `TORCH_HOME` and `HF_HOME` to control where it's cached.

### 2. ES Analysis (PDAnalysis) - Optional

Only needed if `pipeline.es: true`

```bash
# Clone PDAnalysis
git clone https://github.com/mirabdi/PDAnalysis /n/home06/<YOUR_USER>/PDAnalysis
cd /n/home06/<YOUR_USER>/PDAnalysis
pip install -e .
```

Config:
```yaml
es:
  script_dir: /n/home06/<YOUR_USER>/PDAnalysis
  wt_path: /path/to/wildtype_structure.cif  # Your reference structure
  output_dir: /n/home06/<YOUR_USER>/outputs/es
  env_path: /n/home06/<YOUR_USER>/envs/es-analysis  # Optional separate env
```

### 3. SLURM Settings

Update for your account:
```yaml
slurm:
  log_dir: /n/home06/<YOUR_USER>/job_logs
  partition: kempner_requeue  # or your partition
  account: <YOUR_SLURM_ACCOUNT>  # e.g., kempner_<PI>_lab
```

### 4. Input/Output Directories

```yaml
input:
  fasta_dir: /path/to/your/input/fastas

output:
  parent_dir: /n/home06/<YOUR_USER>/outputs
```

## Quick Start Config Template

Copy and modify this for your setup:

```yaml
# config.yaml for new users

pipeline:
  msa: true
  boltz: true
  esm: true
  es: false  # Disable ES initially

input:
  fasta_dir: /n/home06/<YOUR_USER>/data/fastas

output:
  parent_dir: /n/home06/<YOUR_USER>/outputs

# MSA - use shared resources
msa:
  max_files_per_job: 25
  array_max_concurrency: 10
  mmseq2_db: /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db
  colabfold_db: /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/colabfold_db
  colabfold_bin: /n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz/localcolabfold/colabfold-conda/bin

# Boltz - use shared resources
boltz:
  max_files_per_job: 25
  array_max_concurrency: 10
  recycling_steps: 10
  diffusion_samples: 25
  cache_dir: /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db
  colabfold_db: /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/colabfold_db
  env_path: /n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz

# ESM - user setup required
esm:
  num_chunks: 1
  array_max_concurrency: 20
  env_path: /n/home06/<YOUR_USER>/envs/esm
  work_dir: /n/home06/<YOUR_USER>/ProtForge
  cache_dir: /n/home06/<YOUR_USER>/cache/esm

# SLURM
slurm:
  log_dir: /n/home06/<YOUR_USER>/job_logs
  partition: kempner_requeue
  account: <YOUR_SLURM_ACCOUNT>
```

## Recommended: Create Shared ESM Cache

To avoid each user downloading the ESM model separately (~2.4GB), consider:

1. Create shared cache directory:
```bash
mkdir -p /n/holylfs06/LABS/kempner_shared/Everyone/workflow/esm_cache
```

2. Download model once:
```bash
export TORCH_HOME=/n/holylfs06/LABS/kempner_shared/Everyone/workflow/esm_cache
export HF_HOME=/n/holylfs06/LABS/kempner_shared/Everyone/workflow/esm_cache
python -c "from esm.models.esmc import ESMC; ESMC.from_pretrained('esmc_600m')"
```

3. Users set in config:
```yaml
esm:
  cache_dir: /n/holylfs06/LABS/kempner_shared/Everyone/workflow/esm_cache
```

## Directory Structure After Setup

```
/n/holylfs06/LABS/kempner_shared/Everyone/
├── workflow/
│   ├── boltz/
│   │   ├── boltz_db/          # Boltz model weights
│   │   ├── colabfold_db/      # ColabFold database
│   │   └── mmseq2_db/         # MMseqs2 database
│   └── esm_cache/             # TO BE CREATED - shared ESM models
└── common_envs/
    └── miniconda3/envs/boltz/ # Shared Boltz environment

/n/home06/<YOUR_USER>/
├── ProtForge/                 # This repo
├── envs/
│   └── esm/                   # Your ESM conda env
├── cache/
│   └── esm/                   # Your ESM cache (if not using shared)
├── job_logs/                  # SLURM logs
└── outputs/                   # Pipeline outputs
```
