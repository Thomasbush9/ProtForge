# Kempner Cluster Setup Guide

## Quick Start

Run the setup script — it creates shared environments, downloads model weights, patches ESM paths, and generates your `config.yaml`:

```bash
cd /path/to/ProtForge
bash setup.sh
```

The script will prompt for:
- **Shared base directory** — where envs and models live (default: `/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge`)
- **SLURM account** — e.g., `kempner_bsabatini_lab`
- **Email** — for SLURM job notifications (optional)
- **Input FASTA directory** — your input data
- **Output directory** — where results go

After setup, review `config.yaml` and run:
```bash
snakemake --profile profiles/slurm/ -n   # dry run
snakemake --profile profiles/slurm/       # launch
```

## What the Setup Script Does

1. **Creates shared conda environments** at `{shared_base}/envs/`:
   - `esm` — ESM embeddings (esm, torch, numpy)
   - `es-analysis` — ES/PDAnalysis (numpy, scipy, pandas, MDAnalysis)

2. **Downloads ESM model weights** to `{shared_base}/esm_models_cache/`

3. **Patches ESM hardcoded paths** — the ESM library has hardcoded model paths in `esm/utils/constants/esm3.py`. The script updates them to point to the shared cache.

4. **Generates `config.yaml`** with all paths filled in

5. **Validates** that all paths, environments, and tools exist

## Shared Resources (Already Available)

These paths are on the shared filesystem — no setup needed:

### MSA Generation (ColabFold/MMseqs2)
| Config Key | Shared Path |
|------------|-------------|
| `msa.mmseq2_db` | `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db` |
| `msa.colabfold_db` | `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/colabfold_db` |
| `msa.colabfold_bin` | `.../common_envs/miniconda3/envs/boltz/localcolabfold/colabfold-conda/bin` |

### Boltz Structure Prediction
| Config Key | Shared Path |
|------------|-------------|
| `boltz.cache_dir` | `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db` |
| `boltz.env_path` | `/n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz` |

### ESM Embeddings (created by setup.sh)
| Config Key | Shared Path |
|------------|-------------|
| `esm.env_path` | `{shared_base}/envs/esm` |
| `esm.cache_dir` | `{shared_base}/esm_models_cache` |

## Manual Setup (if not using setup.sh)

### 1. ESM Environment

```bash
# Create conda env
conda create -p /path/to/shared/envs/esm python=3.12 -y
/path/to/shared/envs/esm/bin/pip install esm torch numpy

# Download model weights
mkdir -p /path/to/shared/esm_models_cache
TORCH_HOME=/path/to/shared/esm_models_cache \
HF_HOME=/path/to/shared/esm_models_cache \
    /path/to/shared/envs/esm/bin/python -c \
    "from esm.models.esmc import ESMC; ESMC.from_pretrained('esmc_600m')"

# IMPORTANT: Patch hardcoded model paths
# The ESM library hardcodes model paths in:
#   {env}/lib/python3.12/site-packages/esm/utils/constants/esm3.py
# Find lines like:
#   path = Path("/some/old/path/esmc-600m-2024-12")
# Replace with:
#   path = Path("/path/to/shared/esm_models_cache/esmc-600m-2024-12")
```

### 2. ES Analysis (Optional)

```bash
conda create -p /path/to/shared/envs/es-analysis python=3.12 -y
/path/to/shared/envs/es-analysis/bin/pip install numpy scipy pandas MDAnalysis

git clone https://github.com/mirabdi/PDAnalysis /path/to/shared/PDAnalysis
cd /path/to/shared/PDAnalysis
/path/to/shared/envs/es-analysis/bin/pip install -e .
```

### 3. SLURM Settings

```yaml
slurm:
  log_dir: /n/home06/<YOUR_USER>/job_logs
  partition: kempner_requeue
  account: <YOUR_SLURM_ACCOUNT>
  email: <YOUR_EMAIL>
```

### 4. Install Snakemake

```bash
pip install snakemake snakemake-executor-plugin-slurm
```

## Directory Structure After Setup

```
{shared_base}/                          # e.g., /n/holylfs06/LABS/.../protforge
├── envs/
│   ├── esm/                            # Shared ESM conda environment
│   └── es-analysis/                    # Shared ES conda environment
├── esm_models_cache/
│   └── esmc-600m-2024-12/
│       └── data/weights/               # ESM model weights (~2.4GB)
└── PDAnalysis/                         # PDAnalysis repo

/n/holylfs06/LABS/kempner_shared/Everyone/
├── workflow/boltz/
│   ├── boltz_db/                       # Boltz model weights
│   ├── colabfold_db/                   # ColabFold database
│   └── mmseq2_db/                      # MMseqs2 database
└── common_envs/miniconda3/envs/boltz/  # Shared Boltz environment

/n/home06/<YOUR_USER>/
├── ProtForge/                          # This repo (with generated config.yaml)
├── job_logs/                           # SLURM logs
└── outputs/                            # Pipeline results
    ├── sequences/{name}/
    │   ├── {name}.yaml                 # Boltz input
    │   ├── msa/                        # MSA output
    │   ├── boltz/                      # Structure predictions
    │   └── esm/                        # Embeddings + logits
    └── es/                             # ES analysis results
```

## Troubleshooting

### ESM model not found
```
FileNotFoundError: .../esmc_600m_2024_12_v0.pth
```
The ESM library has hardcoded paths. Re-run `bash setup.sh` or manually patch `esm/utils/constants/esm3.py` in the ESM conda env.

### Import error for utils.utils
```
ModuleNotFoundError: No module named 'utils'
```
The Snakemake rule sets `PYTHONPATH` to the repo root. If running manually, do:
```bash
export PYTHONPATH=/path/to/ProtForge
```

### SLURM job OOM / timeout
Increase resources in the rule's `resources:` block in `workflow/rules/*.smk`, or use a non-requeue partition for long jobs.
