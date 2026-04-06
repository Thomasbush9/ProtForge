# ProtForge

A protein structure and function prediction pipeline for SLURM clusters. Orchestrates four ML stages — MSA, Boltz, ESM, and ES — via Snakemake, with automatic chunking, parallel execution, and retry logic.

**Pipeline stages:**
1. **MSA** — Multiple Sequence Alignment (ColabFold/MMseqs2)
2. **Boltz** — Structure prediction (CIF output)
3. **ESM** — Sequence embeddings and logits (ESMC 600M)
4. **ES** — Evolutionary Scale / Effective Strain analysis (PDAnalysis)

Each stage can be toggled on/off independently. Stages can be skipped if you already have intermediate outputs.

## Quick Start (Kempner Cluster)

```bash
# 1. Clone
git clone https://github.com/sabatinilab/ProtForge.git
cd ProtForge

# 2. Setup (generates config.yaml, validates shared resources)
bash setup.sh

# 3. Prepare input data
bash bash_scripts/generate_data.sh --data mutations.tsv --original reference.fasta
# Set input.fasta_dir in config.yaml to the printed path

# 4. Run
snakemake --profile profiles/slurm/ -n    # dry run
snakemake --profile profiles/slurm/        # launch
```

The setup script auto-detects the Kempner cluster and points to shared environments and model weights — no conda installs or downloads needed.

## Quick Start (Other SLURM Clusters)

```bash
git clone https://github.com/sabatinilab/ProtForge.git
cd ProtForge
bash setup.sh --mode custom --shared-base /path/to/shared/dir
```

Custom mode creates conda environments, downloads ESM model weights, and patches hardcoded model paths. You'll need to provide paths to MMseqs2 databases, ColabFold, and Boltz (see [Cluster Setup Guide](docs/CLUSTER_SETUP.md)).

**Requirements:**
- Snakemake 8+ with SLURM executor plugin
- conda/mamba
- SLURM cluster with GPU nodes

```bash
pip install snakemake snakemake-executor-plugin-slurm
```

## Installation

### 1. Clone and setup

```bash
git clone https://github.com/sabatinilab/ProtForge.git
cd ProtForge
bash setup.sh
```

The setup script will:
- Detect your cluster (Kempner vs custom)
- Generate `config.yaml` with correct paths
- Validate all dependencies exist

### 2. Prepare input data

**From a mutation table** (requires reference sequence):
```bash
bash bash_scripts/generate_data.sh --data mutations.tsv --original reference.fasta
```
Input CSV must have an `aaMutations` column (e.g., `SA108D:SN144D`).

**From a name/sequence CSV** (no reference needed):
```bash
bash bash_scripts/generate_data.sh --data sequences.csv
```
Input CSV must have `name` and `sequence` columns.

**Options:**
```bash
--file_type yaml          # Output YAML files (skip MSA, go straight to Boltz)
--subsample N             # Subsample N sequences
--subsample_mode balanced # balanced | fixed | random
```

### 3. Configure

Edit `config.yaml` to toggle stages and set parameters:

```yaml
pipeline:
  msa: true       # Generate MSAs from FASTA
  boltz: true     # Predict structures
  esm: true       # Generate embeddings/logits
  es: false       # Effective strain analysis (needs ref structure)

input:
  fasta_dir: /path/to/fastas    # When running MSA
  # yaml_dir: /path/to/yamls    # When skipping MSA

output:
  parent_dir: /path/to/outputs
```

See `config.template.yaml` for all available parameters.

### 4. Run

```bash
# Full pipeline
snakemake --profile profiles/slurm/

# Dry run (see what would execute)
snakemake --profile profiles/slurm/ -n

# Resume after failure
snakemake --profile profiles/slurm/ --rerun-incomplete

# DAG visualization
snakemake --profile profiles/slurm/ --dag | dot -Tpng > dag.png
```

## Usage Examples

### Run only specific stages

```yaml
# Already have MSA + Boltz outputs, just need ESM embeddings:
pipeline:
  msa: false
  boltz: false
  esm: true
  es: false
```

Then run normally — Snakemake picks up existing outputs from `{output}/sequences/`.

### Multiple Boltz runs per sequence

```yaml
boltz:
  num_runs: 10    # 10 independent structure predictions per sequence
```

Outputs are organized as `sequences/{name}/boltz/run_0/`, `run_1/`, etc.

### Chunking for large datasets

The pipeline automatically splits inputs into chunks for parallel SLURM jobs:

```yaml
msa:
  max_files_per_job: 25     # Sequences per MSA job
boltz:
  max_files_per_job: 25     # YAMLs per Boltz job
esm:
  num_chunks: 4             # Split into 4 parallel ESM jobs
```

## Output Structure

```
{output_dir}/
├── sequences/
│   └── {seq_name}/
│       ├── {seq_name}.yaml       # Boltz input (sequence + MSA path)
│       ├── msa/                  # MSA output (.a3m files)
│       ├── boltz/                # Structure predictions (.cif files)
│       │   ├── run_0/            # (when num_runs > 1)
│       │   └── run_1/
│       └── esm/
│           ├── logits.npy        # Sequence logits
│           └── embeddings.npy    # Sequence embeddings
├── es/                           # ES analysis results (per-residue CSVs)
├── benchmarks/                   # Per-rule timing data
├── benchmark_summary.txt         # Pipeline timing report
├── msa_chunks/                   # Intermediate MSA chunks
├── boltz_chunks/                 # Intermediate Boltz chunks
└── esm_chunks/                   # Intermediate ESM chunks
```

## Architecture

```
Snakefile
├── workflow/rules/msa.smk     chunk_fastas → colabfold_search → scatter_msa
├── workflow/rules/boltz.smk   chunk_yamls → boltz_predict → organize_outputs
├── workflow/rules/esm.smk     chunk_yamls → run_esm (embeddings + logits)
└── workflow/rules/es.smk      collect_cif_paths → PDAnalysis
```

**File format flow:** FASTA → (MSA) → A3M → YAML → (Boltz) → CIF + (ESM) → NPY

**Key design choices:**
- Snakemake checkpoints for dynamic chunking (chunk count determined at runtime)
- `snakemake-executor-plugin-slurm` for native SLURM submission
- Sentinel files (`.msa_complete`, `.boltz_complete`, etc.) for stage dependencies
- Dual-mode execution: Singularity containers or direct `module load` fallback

## Documentation

| Document | Description |
|----------|-------------|
| [Cluster Setup](docs/CLUSTER_SETUP.md) | Environment setup, shared resources, troubleshooting |
| [Snakemake Guide](docs/SNAKEMAKE_GUIDE.md) | How the Snakemake workflow works, SLURM integration |
| [Containers](docs/CONTAINERS.md) | Container design and migration plan |
| [config.template.yaml](config.template.yaml) | Annotated configuration reference |

## Project Structure

```
├── Snakefile                    # Main workflow entry point
├── config.yaml                  # User configuration
├── config.template.yaml         # Annotated config reference
├── setup.sh                     # Cluster setup script
├── workflow/
│   ├── rules/                   # Snakemake rules per stage (.smk)
│   └── scripts/                 # Python helpers (chunking, organizing)
├── slurm_scripts/               # Execution scripts (run_esm.py, etc.)
├── bash_scripts/                # Data preparation (generate_data.sh)
├── utils/                       # Python utilities (mutations, file conversion)
├── profiles/slurm/              # Snakemake SLURM executor profile
└── docs/                        # Documentation
```
