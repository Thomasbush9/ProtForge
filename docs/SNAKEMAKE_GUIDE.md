# Snakemake Pipeline Guide

This guide explains how ProtForge uses Snakemake to orchestrate the protein prediction pipeline on SLURM clusters.

## Quick Reference

```bash
# Install (one-time)
module load python
mamba create -n snakemake -c conda-forge -c bioconda snakemake snakemake-executor-plugin-slurm pyyaml
mamba activate snakemake

# Run full pipeline
snakemake --profile profiles/slurm/

# Dry run (see what would execute, no jobs submitted)
snakemake --profile profiles/slurm/ -n

# Resume after failure (re-runs only what's missing)
snakemake --profile profiles/slurm/ --rerun-incomplete

# Visualize the DAG
snakemake --profile profiles/slurm/ --dag | dot -Tpng > dag.png

# Check status summary
snakemake --profile profiles/slurm/ --summary
```

## How Snakemake Works

### Core Concepts

**Rules** define steps in the pipeline. Each rule has inputs, outputs, and a shell command:

```python
rule my_step:
    input: "data/input.txt"
    output: "results/output.txt"
    shell: "process {input} > {output}"
```

**DAG (Directed Acyclic Graph):** Snakemake builds a dependency graph from rules. It figures out which rules need to run based on what output files are missing. If an output already exists, that step is skipped.

**Checkpoints** are special rules whose outputs determine what downstream jobs to create. We use these for chunking — the number of chunks isn't known until the checkpoint runs.

**Wildcards** (`{chunk_id}`, `{run_id}`) let a single rule definition create many jobs. Snakemake expands them into concrete jobs at runtime.

**Localrules** run on the login node instead of being submitted as SLURM jobs. Used for lightweight tasks like file splitting and organizing outputs.

### How Snakemake + SLURM Work Together

Without Snakemake, you'd manually write `sbatch` scripts, chain them with `--dependency=afterok:JOBID`, and handle failures yourself. Snakemake automates all of this:

1. **Job submission:** The `snakemake-executor-plugin-slurm` plugin translates each rule into an `sbatch` call with the right resources (GPUs, memory, time, partition).

2. **Dependencies:** Snakemake tracks which jobs depend on which outputs. It only submits a job once its inputs exist.

3. **Retries:** Failed jobs are automatically retried (configured via `restart-times` in the profile). On `kempner_requeue`, jobs can be preempted — retries handle this transparently.

4. **Resumability:** If the pipeline crashes partway through, re-running the same command picks up where it left off. Completed steps (whose output files exist) are skipped.

### SLURM Profile

The file `profiles/slurm/config.yaml` configures the executor:

```yaml
executor: slurm              # Use SLURM for job submission
jobs: 100                    # Max concurrent SLURM jobs
latency-wait: 60             # Seconds to wait for output files on NFS
restart-times: 2             # Retry failed jobs up to 2 times
max-status-checks-per-second: 1
local-cores: 4               # Cores available for localrules
```

Per-rule resources (GPUs, memory, partition) are defined in the `.smk` rule files, not in the profile.

## ProtForge Pipeline Stages

```
FASTA files
    |
    v
[chunk_fastas]          <- localrule: splits FASTAs into chunks
    |
    v
[run_colabfold_search]  <- SLURM job: GPU, 48GB RAM, 1 per chunk
    |
    v
[scatter_msa]           <- localrule: organizes MSA outputs, creates YAMLs
    |
    v
[chunk_yamls_for_boltz] <- localrule: splits YAMLs into chunks
    |
    v
[run_boltz_predict]     <- SLURM job: GPU, 16GB RAM, 1 per (chunk x run)
    |
    v
[organize_boltz_chunk]  <- localrule: copies final model to seq dir
    |
    v
Done: sequences/{seq}/boltz/run_N/{seq}_model_24.cif
```

### Stage toggles

In `config.yaml`, each stage can be enabled/disabled:

```yaml
pipeline:
  msa: true      # MSA generation (ColabFold)
  boltz: true    # Structure prediction (Boltz)
  esm: false     # Embedding extraction (ESM-C)
  esmfold: false # Structure prediction (ESMFold — alternative / complement to Boltz)
  es: false      # Evolutionary scale analysis (PDAnalysis)
```

### Chunking and Parallelism

Large datasets are split into chunks for parallel processing:

1. **chunk_fastas** splits input FASTAs into directories of N files each (default: 25).
2. Each chunk becomes a separate SLURM job.
3. With 100 sequences and `max_files_per_job: 25`, you get 4 parallel GPU jobs.

The `checkpoint` mechanism makes this dynamic — Snakemake doesn't know how many chunks exist until the chunking rule runs.

### Multi-Run Boltz Predictions

When `boltz.num_runs > 1`, each chunk is run N times independently:

```yaml
boltz:
  num_runs: 10           # 10 independent Boltz runs per sequence
  diffusion_samples: 25  # 25 diffusion steps per run
```

This creates a `(chunk x run)` matrix of SLURM jobs, all submitted in parallel. For 4 chunks x 10 runs = 40 GPU jobs. Only the final model (`model_24`) from each run is kept.

Output structure:
```
sequences/{seq}/boltz/
    run_0/{seq}_model_24.cif
    run_1/{seq}_model_24.cif
    ...
    run_9/{seq}_model_24.cif
```

When `num_runs: 1` (default), the `run_N/` subdirectory is omitted for backward compatibility.

## File Layout

```
ProtForge/
├── Snakefile                  # Entry point — includes stage rules, defines targets
├── config.yaml                # Your pipeline configuration
├── profiles/slurm/
│   └── config.yaml            # Snakemake SLURM executor settings
└── workflow/
    ├── rules/
    │   ├── msa.smk            # MSA stage rules
    │   ├── boltz.smk          # Boltz stage rules
    │   ├── esm.smk            # ESM stage rules
    │   ├── esmfold.smk        # ESMFold stage rules
    │   └── es.smk             # ES stage rules
    └── scripts/
        ├── chunk_fastas.py           # Split FASTAs into chunks
        ├── organize_msa_outputs.py   # Scatter a3m files, create YAMLs
        ├── prepare_boltz_chunks.py   # Symlink YAMLs into chunk dirs
        ├── organize_boltz_outputs.py # Copy best model to seq dir
        └── chunk_yamls_for_esm.py    # Split YAML or FASTA paths (ESM + ESMFold)
```

## Sentinel Files

Snakemake tracks completion via sentinel (marker) files:

| File | Meaning |
|------|---------|
| `.msa_complete` | All MSA chunks processed and scattered |
| `.boltz_complete` | All Boltz runs organized |
| `.esm_complete` | All ESM embeddings extracted |
| `.esmfold_complete` | All ESMFold structures predicted |
| `es/.done` | ES analysis complete |
| `chunk_X_run_Y_output/.done` | One Boltz prediction job finished |

These are empty files created by `touch` at the end of each rule. If a sentinel exists, Snakemake skips that step.

## Common Operations

### Run only MSA (skip Boltz)

Set `pipeline.boltz: false` in `config.yaml`, then run normally.

### Run Boltz on existing YAMLs (skip MSA)

```yaml
pipeline:
  msa: false
  boltz: true
input:
  yaml_dir: /path/to/existing/yamls
```

### Run Boltz multi-run standalone (after MSA)

```bash
./run_boltz_predictions.sh /path/to/outputs/sequences 10
```

This submits SLURM jobs directly (without Snakemake) for N independent Boltz runs.

### Run ESMFold

ESMFold (`facebook/esmfold_v1` via HuggingFace) predicts structures from sequence only — no MSA required. It writes one PDB per sequence to `{output}/sequences/{seq}/esmfold/structure.pdb` plus a `plddt.npy` array.

**One-time model download** (on a login node, needs internet):

```bash
conda activate /n/holylfs06/LABS/bsabatini_lab/Everyone/protforge/envs/esmfold
python scripts/download_models.py --models esmfold \
    --cache-dir /n/holylfs06/LABS/bsabatini_lab/Everyone/esm_models_cache
```

This populates `<cache_dir>/hub/models--facebook--esmfold_v1/` (~22 GB). The SLURM worker loads from this cache offline, so compute nodes don't need internet.

**Config:**

```yaml
pipeline:
  esmfold: true
esmfold:
  input_type: yaml            # or "fasta" — see below
  num_chunks: 4               # parallel GPU jobs
  env_path: /n/holylfs06/LABS/bsabatini_lab/Everyone/protforge/envs/esmfold
  cache_dir: /n/holylfs06/LABS/bsabatini_lab/Everyone/esm_models_cache
```

**Two input modes:**

| `input_type` | Source dir | Glob | Upstream dependency |
|---|---|---|---|
| `yaml` *(default)* | `input.yaml_dir` if set, else `{output}/sequences/` | `*.yaml` | Waits for MSA/Boltz completion |
| `fasta` | `input.fasta_dir` | `*.fasta` | None — runs independently |

Use `fasta` mode when you want to fold raw sequences without MSA generation (fastest path: ~1–2 min model load + ~5–30 s per sequence on an H100).

**Run it:**

```bash
snakemake --profile profiles/slurm/
```

### Force re-run of a specific stage

Delete its sentinel file and re-run:

```bash
rm /path/to/outputs/.boltz_complete
snakemake --profile profiles/slurm/
```

### Clean up and start fresh

```bash
# Remove snakemake metadata (locks, temp files)
snakemake --profile profiles/slurm/ --cleanup-metadata

# Or if locked after a crash
snakemake --profile profiles/slurm/ --unlock
```

## Troubleshooting

### "MissingInputException" at DAG time

The input directory doesn't exist yet. If MSA creates it, make sure `pipeline.msa: true` so Snakemake knows to create it first.

### "kempner_requeue requires GPU"

Lightweight rules (chunking, organizing) must be `localrule`s so they run on the login node, not on GPU partitions. This is already configured in the current rules.

### Jobs fail with "Triton" errors

You got assigned a MIG GPU (partitioned A100 with reduced memory). Re-run and hope for a full GPU:

```bash
snakemake --profile profiles/slurm/ --rerun-incomplete
```

Or switch to the non-requeue partition in config:

```yaml
slurm:
  boltz:
    partition: kempner  # guaranteed full GPUs, but lower priority
```

### Pipeline is "locked"

Another Snakemake process is running, or a previous run crashed without cleaning up:

```bash
snakemake --profile profiles/slurm/ --unlock
```

### SLURM job logs

Snakemake stores SLURM logs at:

```
.snakemake/slurm_logs/rule_{rule_name}/{chunk_id}_{run_id}/{slurm_job_id}.log
```

Check these for the actual error output from `boltz predict`, `colabfold_search`, etc.

### Monitor running jobs

```bash
squeue -u $USER                    # All your jobs
squeue -u $USER --name=snakemake   # Snakemake-submitted jobs
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed  # Job details
```

## Key Differences from Legacy Bash Pipeline

| | Snakemake (`snakemake` branch) | Bash (`main` branch) |
|---|---|---|
| Job submission | Automatic via plugin | Manual `sbatch` chains |
| Dependencies | DAG-based, automatic | `--dependency=afterok:JOBID` |
| Retries | Built-in (`restart-times`) | Checker scripts (`checker.sh`) |
| Resume | Re-run same command | Re-run same command |
| Parallelism | Automatic (up to `jobs` limit) | SLURM array jobs |
| Multi-run Boltz | `num_runs` config param | Not supported |
| Config | Single `config.yaml` | Environment variables + config |
