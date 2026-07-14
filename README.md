# ProtForge

A protein structure and function prediction pipeline for SLURM clusters.
Orchestrates several ML stages via Snakemake, with automatic chunking, parallel
SLURM execution, calibrated resource estimation, and resume-on-failure.

**Pipeline stages** (each toggled on/off independently in `config.yaml`):
1. **MSA** — Multiple Sequence Alignment (ColabFold / MMseqs2)
2. **Boltz** — Structure prediction from the MSA (CIF output)
3. **ESM-C** — Sequence embeddings / logits (300M / 600M / 6B); optional SAE sparse-activation extraction
4. **ESMFold** — ESMFold2 structure prediction (sequence-only)
5. **OpenFold** — OpenFold3 structure prediction from the MSA

ESM-C and ESMFold run straight from sequence (no MSA), in parallel with the
MSA → Boltz / OpenFold branch. Each GPU stage runs inside a Singularity/Apptainer
container image.

---

## Easiest path: drive ProtForge with Claude Code

ProtForge ships a suite of [Claude Code](https://claude.com/claude-code) **skills**
that do the install, sizing, launch, monitoring, and analysis for you — you talk
to them in plain English instead of memorizing flags. This is the recommended way
to get started.

### 1. Install Claude Code

```bash
# Node 18+ required
npm install -g @anthropic-ai/claude-code
```

(Alternatively, the native installer: `curl -fsSL https://claude.ai/install.sh | bash`.)

### 2. Launch it in the repo

```bash
git clone https://github.com/Thomasbush9/ProtForge.git
cd ProtForge
claude
```

On first run, `claude` walks you through login. Once inside, the ProtForge skills
auto-load (they live in `.claude/skills/`).

### 3. Ask for what you want

Type a slash command, or just describe the task and Claude picks the right skill:

| Skill | Invoke with `/<name>` or just ask… | What it does |
|-------|-----------------------------------|--------------|
| **setup** | "set up ProtForge" | First-time install: workspace dirs, `config.yaml`, container build, model-weight download, smoke test |
| **fetch-sequences** | "fetch these UniProt accessions" | Turns an `.xlsx`/`.csv` of accessions into a per-protein FASTA dir |
| **run-pipeline** | "run the pipeline on these FASTAs" | Sizes SLURM resources from the calibrated models, applies them, dry-runs, submits, monitors |
| **tune-params** | "best Boltz settings for a fast screen?" | Advises stage parameters for a speed/quality goal |
| **monitor** | "why did my run fail?" / "what's stuck?" | Read-only status, failure triage, recovery guidance |
| **results** | "summarize the run" | Per-stage runtime/memory + per-structure confidences (pLDDT/pTM/ipTM) |
| **satmut** | "saturation-mutagenesis scan of GFP" | Zero-shot ESM-C variant-effect (LLR) scan + heatmap |

A typical first session is just: **"set up ProtForge"** → **"run the pipeline on
`/path/to/fastas`"** → **"summarize the run"**.

> Prefer to drive it by hand? The manual workflow is below — the skills are a
> convenience layer over exactly these scripts, not a replacement.

---

## Manual setup (Kempner cluster)

There is no `setup.sh` for the container path. Setup is: build (or reuse) the GPU
container image, download model weights, and fill `config.yaml`. See the
[Cluster Setup Guide](docs/CLUSTER_SETUP.md) for the full first-time walkthrough.

```bash
# 1. Clone
git clone https://github.com/Thomasbush9/ProtForge.git
cd ProtForge

# 2. Build the per-stage GPU containers (on a compute node — needs --fakeroot).
#    ProtForge uses one image per stage; `all` builds msa/boltz/esm/openfold.
export PROTFORGE_ROOT=/n/holylfs06/LABS/<your_lab>/Everyone/<you>   # SIFs land in $PROTFORGE_ROOT/sifs
bash containers/build.sh all
#   or build a subset:  bash containers/build.sh boltz esm
#   or pull one prebuilt image:  bash containers/build.sh boltz --from-docker docker://ghcr.io/<owner>/protforge-boltz:latest

# 3. Download ESM-C / ESMFold weights to a host HF cache
python scripts/download_models.py --cache-dir "$PROTFORGE_ROOT/models/hf"

# 4. Create your config. On Kempner, start from the Kempner template — the shared
#    DBs, partition, container runtime and $PROTFORGE_ROOT-relative SIF/cache
#    paths are already filled in.
cp config.kempner.template.yaml config.yaml
#   edit only: slurm.account, slurm.email, input.fasta_dir
#   (keep PROTFORGE_ROOT exported — the config expands ${PROTFORGE_ROOT} at load time)
#
#   Not on Kempner, or want the full annotated parameter reference?
#   cp config.template.yaml config.yaml   # then fill in every path by hand
```

**Shared resources already on Kempner** (no setup needed — leave the template
defaults): the MSA databases (`msa.mmseq2_db`, `msa.colabfold_db`) and the Boltz
checkpoint (`boltz.cache_dir`).

> **Container cache paths:** `esmc.cache_dir` / `esmfold.cache_dir` /
> `openfold.cache_dir` are **host** directories — the rules bind them read-only
> into the container at `/models/hf` (and `/models/openfold`). Don't set them to
> the in-container path.

**Requirements:**
- Snakemake 8+ with the SLURM executor plugin, in a conda/mamba env
- Singularity/Apptainer + a SLURM cluster with GPU nodes

```bash
pip install snakemake snakemake-executor-plugin-slurm
```

## Prepare input data

**From a mutation table** (requires a reference sequence):
```bash
bash bash_scripts/generate_data.sh --data mutations.tsv --original reference.fasta
```
Input CSV must have an `aaMutations` column (e.g., `SA108D:SN144D`).

**From a name/sequence CSV** (no reference needed):
```bash
bash bash_scripts/generate_data.sh --data sequences.csv
```
Input CSV must have `name` and `sequence` columns.

**From a spreadsheet of UniProt accessions:** use the `fetch-sequences` skill, or
`scripts/uniprot_fetch/fetch_sequences.py` directly (see its README).

**Options:**
```bash
--file_type yaml           # Output YAML files (skip MSA, go straight to Boltz)
--subsample N              # Subsample N sequences
--subsample_mode balanced  # balanced | fixed | random
```

Set the printed output directory as `input.fasta_dir` (or `input.yaml_dir`) in
`config.yaml`.

## Configure

Edit `config.yaml` to toggle stages and set parameters:

```yaml
pipeline:
  msa: true        # Generate MSAs from FASTA
  boltz: true      # Boltz structure prediction
  esmc: true       # ESM-C embeddings/logits
  esmfold: true    # ESMFold2 structure prediction
  openfold: false  # OpenFold3 structure prediction

input:
  fasta_dir: /path/to/fastas    # When running MSA / sequence stages
  # yaml_dir: /path/to/yamls    # When skipping MSA

output:
  parent_dir: /path/to/outputs
```

See [config.template.yaml](config.template.yaml) for every annotated parameter.
Per-stage SLURM resources (`slurm.resources.*`) are normally written by the
estimator (`python -m webapp.estimate_cli --config config.yaml --apply`), not by
hand.

## Run

```bash
# Dry run (see what would execute)
snakemake --profile profiles/slurm/ -n

# Full pipeline
snakemake --profile profiles/slurm/

# Resume after a failure / preemption (picks up where it left off)
snakemake --profile profiles/slurm/ --rerun-incomplete

# DAG visualization
snakemake --profile profiles/slurm/ --dag | dot -Tpng > dag.png
```

## Web UI

A Streamlit front-end exposes config editing, the resource estimator, live
monitoring, results, and saturation-mutagenesis:

```bash
streamlit run webapp/app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
# then tunnel from your laptop:  ssh -L 8501:localhost:8501 <user>@<login-node>
```

See the [Web UI guide](docs/WEBAPP.md) for SSH / VS Code / Open OnDemand access.

## Usage Examples

### Run only specific stages

```yaml
# Already have MSAs; just want ESM-C embeddings + ESMFold structures:
pipeline:
  msa: false
  boltz: false
  esmc: true
  esmfold: true
  openfold: false
```

Snakemake picks up existing outputs from `{output}/sequences/`.

### Multiple Boltz runs per sequence

```yaml
boltz:
  num_runs: 10    # 10 independent structure predictions per sequence
```

### Chunking for large datasets

The pipeline splits inputs into chunks for parallel SLURM jobs:

```yaml
msa:
  max_files_per_job: 50     # Sequences per MSA job
boltz:
  max_files_per_job: 25     # YAMLs per Boltz job
esmc:
  max_files_per_job: 25     # Sequences per ESM-C job
```

## Output Structure

```
{output_dir}/
├── sequences/
│   └── {seq_name}/
│       ├── {seq_name}.yaml          # Boltz input (sequence + MSA path)
│       ├── msa/                     # MSA output (.a3m files)
│       ├── boltz/                   # Boltz structures (.cif) + confidence JSONs
│       ├── openfold/                # OpenFold3 structures (.cif) + confidence JSONs
│       ├── esmfold/fast/            # ESMFold2 structure.cif, plddt.npy, metrics.pt
│       └── esmc/{size}/             # ESM-C embeddings/logits (+ sae/{size}/ if enabled)
├── benchmarks/                      # Per-rule timing data (TSV)
├── benchmark_summary.txt            # Pipeline timing report
├── logs/                            # Per-stage logs
├── msa_chunks/  boltz_chunks/  esmc_chunks/  esmfold_chunks/  openfold_chunks/
└── ...                              # Intermediate chunk manifests
```

## Architecture

```
Snakefile
├── workflow/rules/msa.smk        chunk_fastas → colabfold_search → scatter_msa
├── workflow/rules/boltz.smk      chunk_yamls → boltz_predict → organize_outputs
├── workflow/rules/esmc.smk       chunk_yamls → run_esm (embeddings + logits)
├── workflow/rules/esmc_sae.smk   SAE sparse-activation extraction (optional)
├── workflow/rules/esmfold.smk    chunk_yamls → ESMFold2 fold
└── workflow/rules/openfold.smk   chunk_yamls → OpenFold3 predict → organize
```

**File-format flow:** FASTA → (MSA) → A3M → YAML → (Boltz / OpenFold) → CIF;
sequence → (ESM-C) → NPY, (ESMFold) → CIF.

**Key design choices:**
- Snakemake checkpoints for dynamic chunking (chunk count determined at runtime)
- `snakemake-executor-plugin-slurm` for native SLURM submission
- Sentinel files (`.msa_complete`, `.boltz_complete`, …) for stage dependencies
- Each GPU stage runs in its own container image (per-stage SIF, with a shared
  fallback); host caches/DBs are bind-mounted per rule

## Documentation

| Document | Description |
|----------|-------------|
| [Cluster Setup](docs/CLUSTER_SETUP.md) | Environment setup, shared resources, troubleshooting |
| [Web UI](docs/WEBAPP.md) | Streamlit front-end: install + access (SSH / VS Code / Open OnDemand) |
| [Snakemake Guide](docs/SNAKEMAKE_GUIDE.md) | How the Snakemake workflow works, SLURM integration |
| [Containers](containers/README.md) | Per-stage container design and build/pull instructions |
| [config.kempner.template.yaml](config.kempner.template.yaml) | Ready-to-run Kempner config — set account + email and go |
| [config.template.yaml](config.template.yaml) | Cluster-agnostic annotated configuration reference |

## Project Structure

```
├── Snakefile                    # Main workflow entry point
├── config.kempner.template.yaml # Kempner-ready config (copy → config.yaml)
├── config.template.yaml         # Annotated config reference (copy → config.yaml)
├── containers/                  # Container defs + build.sh + per-stage test scripts
├── workflow/
│   ├── rules/                   # Snakemake rules per stage (.smk)
│   └── scripts/                 # Python helpers (chunking, organizing)
├── webapp/                      # Streamlit UI + estimator / monitor / results / satmut CLIs
├── scripts/                     # download_models.py, uniprot_fetch/, calibration
├── bash_scripts/                # Data preparation (generate_data.sh)
├── utils/                       # Python utilities (mutations, file conversion)
├── profiles/slurm/              # Snakemake SLURM executor profile
├── .claude/skills/              # Claude Code skill suite (setup, run-pipeline, …)
└── docs/                        # Documentation
```
