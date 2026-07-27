# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Lab notes live in the vault

Experimental logs, design decisions, and calibration findings are in
`~/Documents/Vault/Notes/Lab/protforge/`, not in this repo. When the user
opens a session via `cclab protforge`, that directory autoloads.

- `Lab/protforge/agenda.md` — current focus / TODOs
- `Lab/protforge/log/YYYY-MM-DD-*.md` — per-experiment writeups
- `Lab/protforge/calibration.md` — durable reference for the resource-calibration workflow
- `Lab/protforge/decisions.md` — design choices and rationale
- `Lab/protforge/data.md` — cluster paths, run IDs

The repo holds code, runtime configs, and operational docs only
(`docs/SNAKEMAKE_GUIDE.md`, `docs/CLUSTER_SETUP.md`, `containers/README.md`).
Don't add new long-form `*.md` to the repo — write to the vault instead.

## Project Overview

ProtForge is a protein structure and function prediction pipeline designed for the Kempner cluster. It orchestrates a multi-stage ML pipeline:
1. **MSA** - Multiple Sequence Alignment generation (ColabFold/MMseq2)
2. **Boltz** - Structure prediction
3. **ESM** - Embedding/logits generation
4. **ES** - Evolutionary Scale analysis (PDAnalysis)

## Common Commands

### Snakemake workflow (recommended, `snakemake` branch)
```bash
# Full pipeline (all enabled stages)
snakemake --profile profiles/slurm/

# Dry run (see what would execute)
snakemake --profile profiles/slurm/ -n

# DAG visualization
snakemake --profile profiles/slurm/ --dag | dot -Tpng > dag.png

# Resume after failure (just re-run — picks up where it left off)
snakemake --profile profiles/slurm/ --rerun-incomplete

# Run specific stage only
snakemake --profile profiles/slurm/ {OUTPUT}/.boltz_complete

# Check status
snakemake --profile profiles/slurm/ --summary
```

Requires: `pip install snakemake snakemake-executor-plugin-slurm`

### Bash pipeline (`main` branch fallback)

#### Run the core pipeline (MSA + Boltz)
```bash
./run.sh [CONFIG_FILE]           # defaults to config.yaml
```
`run.sh` runs MSA and Boltz only. ESM and ES are run separately via standalone scripts.

#### Run individual stages (standalone pipelines)
```bash
./run_msa.sh [CONFIG_FILE]                     # MSA only (from FASTA files)
./run_boltz.sh YAML_DIR [CONFIG_FILE]          # Boltz only (from existing YAML)
./run_esm_standalone.sh YAML_DIR [CONFIG_FILE] # ESM only (from existing YAML)
./run_es_standalone.sh CIF_DIR [CONFIG_FILE]   # ES only (from existing CIF files)
```

### Prepare data from CSV
```bash
# From mutation table (requires reference sequence)
bash bash_scripts/generate_data.sh --data mutations.tsv --original reference.fasta

# From name/sequence CSV (no reference needed)
bash bash_scripts/generate_data.sh --data sequences.csv

# Output YAML to skip MSA generation
bash bash_scripts/generate_data.sh --data sequences.csv --file_type yaml

# Optional: --subsample N --subsample_mode balanced|fixed|random
```

Input CSV formats:
- **Mutations mode**: CSV with `aaMutations` column (e.g., `SA123G:SB456T`)
- **Sequences mode**: CSV with `name` and `sequence` columns

### Install tools (container image + model weights)
```bash
# Build (or pull) the GPU container, then download ESM-C / ESMFold weights:
bash containers/build.sh                                  # or: --from-docker docker://...
python scripts/download_models.py --cache-dir "$PROTFORGE_ASSETS/models/hf"
```
See `docs/CLUSTER_SETUP.md` for the full first-time setup.

### Check for errors and retry failed jobs
```bash
./slurm_scripts/checker.sh msa|boltz|esm <output_dir> [config.yaml]
```

### Monitor running jobs
```bash
squeue -u $USER
```

## Architecture

### Configuration-Driven Pipeline
All parameters are in `config.yaml`. Pipeline stages can be toggled on/off via `pipeline.msa`, `pipeline.boltz`, etc.

### Snakemake Workflow (`snakemake` branch)
```
Snakefile (rule all)
├─→ workflow/rules/msa.smk   (chunk_fastas → colabfold_search → scatter_msa)
├─→ workflow/rules/boltz.smk (chunk_yamls → boltz_predict → organize_outputs)
├─→ workflow/rules/esm.smk   (chunk_yamls → run_esm.py)
└─→ workflow/rules/es.smk    (build_cif_paths → MPI PDAnalysis)
```

Uses Snakemake checkpoints for dynamic chunking, `snakemake-executor-plugin-slurm` for SLURM submission, and built-in retries (no manual checker scripts needed).

### Bash Orchestration Flow (`main` branch)
```
run.sh (MSA + Boltz)
├─→ split_and_run_msa.sh → run_msa_array.slrm → process_msa_fasta.sh
└─→ run_boltz_wrapper.slrm → split_and_run_boltz.sh → run_boltz_array.slrm

run_esm_standalone.sh → run_esm.sh → run_esm_array.slrm → run_esm.py
run_es_standalone.sh  → run_es.sh  → run_es_array.slrm
```

Dependencies use SLURM's `afterok` for sequential execution. Checker jobs run on `afternotok` for error recovery.

### Key Directories
- `workflow/rules/` - Snakemake rule files per stage (.smk)
- `workflow/scripts/` - Python helper scripts for chunking and output organization
- `profiles/slurm/` - Snakemake SLURM executor profile
- `slurm_scripts/` - Legacy SLURM job templates (.slrm), orchestration scripts, checkers
- `bash_scripts/` - Data preparation
- `utils/` - Python utilities for mutation generation, file format conversion

### File Format Flow
FASTA → (MSA) → A3M → YAML (Boltz input) → CIF (structure output)

### Chunking Pattern
Large datasets are split into chunks (e.g., `id_0.txt`, `id_1.txt`) listing file paths. SLURM array jobs process chunks in parallel using `filelist.manifest` for indexing.

### Progress Tracking
- `processed_paths.txt` - Successfully completed files
- `total_paths.txt` - All files to process
- Enables retry logic in checker scripts

## Config Structure

```yaml
pipeline:
  msa: true|false
  boltz: true|false
  esm: true|false
  es: true|false

input:
  fasta_dir: /path/to/fastas     # when running MSA
  yaml_dir: /path/to/yamls       # when skipping MSA

output:
  parent_dir: /path/to/outputs

msa/boltz/esm/es:
  # stage-specific settings (chunk sizes, cache dirs, env paths)

slurm:
  partition: default_partition
  account: account_name
  log_dir: /path/to/logs
  # per-job partition overrides: msa:, boltz:, esm:, checker_msa:, etc.
```

## Dependencies

Python dependencies for data generation: `requirements-data.txt` (pandas, numpy, PyYAML, tqdm)

External tools are provided by the GPU container image (built or pulled via
`containers/build.sh`); model weights are fetched separately with
`scripts/download_models.py --cache-dir <HF_CACHE>`:
- Boltz (structure prediction)
- ESM-C / ESMFold (embeddings, logits, structure prediction)
- PDAnalysis (ES analysis)

## Cluster Setup

See `docs/CLUSTER_SETUP.md` for Kempner cluster configuration.

**Shared resources** (no setup needed):
- MSA databases (MMseqs2, ColabFold)
- Boltz model weights and conda environment

**User setup required**:
- ESM conda environment and model cache
- ES/PDAnalysis (optional)
- SLURM account settings

Copy `config.template.yaml` to `config.yaml` and update paths.
