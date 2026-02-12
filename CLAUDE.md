# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProtForge is a protein structure and function prediction pipeline designed for the Kempner cluster. It orchestrates a multi-stage ML pipeline:
1. **MSA** - Multiple Sequence Alignment generation (ColabFold/MMseq2)
2. **Boltz** - Structure prediction
3. **ESM** - Embedding/logits generation
4. **ES** - Evolutionary Scale analysis (PDAnalysis)

## Common Commands

### Run the full pipeline
```bash
./run.sh [CONFIG_FILE]           # defaults to config.yaml
```

### Run individual stages (standalone pipelines)
```bash
./run_msa.sh [CONFIG_FILE]                    # MSA only (from FASTA files)
./run_boltz.sh YAML_DIR [CONFIG_FILE]         # Boltz only (from existing YAML)
./run_esm_standalone.sh YAML_DIR [CONFIG_FILE] # ESM only (from existing YAML)
```

### Prepare data from CSV
```bash
# From mutation table (requires reference sequence)
bash bash_scripts/generate_data.sh --data mutations.tsv --original reference.fasta

# From name/sequence CSV (no reference needed)
bash bash_scripts/generate_data.sh --data sequences.csv

# Output YAML to skip MSA generation
bash bash_scripts/generate_data.sh --data sequences.csv --file_type yaml

# Optional: --subsample N --subsample_mode balanced|fixed
```

Input CSV formats:
- **Mutations mode**: CSV with `aaMutations` column (e.g., `SA123G:SB456T`)
- **Sequences mode**: CSV with `name` and `sequence` columns

### Install tools (Boltz, ESM, ES)
```bash
bash download_tools.sh [--cache-dir DIR] [--config config.yaml] boltz esm es
```

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
All parameters are in `config.yaml`. The `slurm_scripts/parse_config.py` utility extracts values for bash scripts. Pipeline stages can be toggled on/off via `pipeline.msa`, `pipeline.boltz`, etc.

### Job Orchestration Flow
```
run.sh
├─→ split_and_run_msa.sh → run_msa_array.slrm → process_msa_fasta.sh
├─→ run_boltz_wrapper.slrm → split_and_run_boltz.sh → run_boltz_array.slrm
├─→ run_esm_wrapper.slrm → run_esm.sh → run_esm_array.slrm → run_esm.py
└─→ run_es.sh
```

Dependencies use SLURM's `afterok` for sequential execution. Checker jobs run on `afternotok` for error recovery.

### Key Directories
- `slurm_scripts/` - SLURM job templates (.slrm), orchestration scripts, checkers
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

External tools installed via `download_tools.sh`:
- Boltz (structure prediction)
- ESM (embeddings via fair-esm)
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
