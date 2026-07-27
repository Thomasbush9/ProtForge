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
> convenience layer over exactly these scripts, not a replacement. For a
> point-and-click alternative, jump to [Web UI](#web-ui): a first-time setup and
> a browser tab, with no `config.yaml` to write.

---

## Setup (Kempner)

The container images and model weights already exist on the cluster, so there is
nothing to build or download. Setup is two exports, a clone, and an environment.

```bash
# 1. Where the images and weights live, and where YOUR runs land.
export PROTFORGE_ASSETS=/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge-assets
export PROTFORGE_ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/<you>
#    Add both to ~/.bashrc so every shell has them.

# 2. Clone.
git clone https://github.com/Thomasbush9/ProtForge.git "$PROTFORGE_ROOT/ProtForge"
cd "$PROTFORGE_ROOT/ProtForge"

# 3. Activate the shared host environment (Snakemake + the web UI).
#    Nothing to install — it lives beside the images.
module load python && mamba activate "$PROTFORGE_ASSETS/envs/host"
```

That's it. Nothing is copied to your workspace: `PROTFORGE_ASSETS` is read-only
and shared, `PROTFORGE_ROOT` holds only your inputs, outputs and job logs.

> `PROTFORGE_ASSETS` above is readable by the **bsabatini lab**. On Kempner
> outside the lab, see [Building your own images](#building-your-own-images).

### The host environment

`$PROTFORGE_ASSETS/envs/host` is a shared conda environment holding Snakemake,
the SLURM executor plugin, Streamlit and the Results tab's viewer/plot deps.
Nothing heavier lives there — PyTorch, ESM, Boltz and OpenFold are all inside
the container images. Activating it is the whole install.

**It must be *activated* every time you run `snakemake` or the web UI.** The
workflow's chunking rules run locally and shell out to a bare `python`, so an
unactivated shell fails on the very first rule with `python: command not found`,
before a single cluster job is submitted.

On Kempner, `mamba` only exists as a shell function *after* `module load python`,
and it does not survive into a non-interactive shell — chain both in one
invocation. Put this in your `~/.bashrc` alongside the two exports:

```bash
module load python && mamba activate "$PROTFORGE_ASSETS/envs/host"
python -c "import snakemake"     # sanity check: should print nothing
```

Read-only, so you can't `pip install` into it. If you need extra packages, or
can't read the shared copy, build your own — a conda env or a plain venv both
work, the only requirement being that `python` and `snakemake` are on `PATH`:

```bash
module load python
mamba create -p "$PROTFORGE_ROOT/envs/host" python=3.13 -y
mamba activate "$PROTFORGE_ROOT/envs/host"
pip install -r requirements-host.txt
```

Put it on lab storage as shown, not in `~` or `~/.conda` — home quotas are small
and a host env is a couple of GB. Note that a conda env cannot be relocated by
copying (absolute prefixes are baked into its shebangs); use
`mamba create -p <new> --clone <old>` if you need it somewhere else.

Then either launch the [Web UI](#web-ui) — no config file to write — or work from
the command line:

```bash
module load python && mamba activate "$PROTFORGE_ASSETS/envs/host"
cp config.kempner.template.yaml config.yaml
$EDITOR config.yaml     # set slurm.account, slurm.email, input.fasta_dir
snakemake --profile profiles/slurm/ -n      # dry run
snakemake --profile profiles/slurm/         # go
```

The MSA databases, the Boltz checkpoint, the partition and the container runtime
are already correct in that template — leave them alone.

### Building your own images

Only if you can't read a shared `PROTFORGE_ASSETS`. Point it at your own
workspace, then build once (on a compute node — `--fakeroot` is not permitted on
login nodes) and fetch the weights:

```bash
export PROTFORGE_ASSETS="$PROTFORGE_ROOT"
salloc -p test --account=<your_slurm_account> -t 4:00:00 --mem 32G --ntasks-per-node 4
bash containers/build.sh all        # or a subset: bash containers/build.sh msa boltz
exit
python scripts/download_models.py --cache-dir "$PROTFORGE_ASSETS/models/hf"
```

Expect ~1 h and ~120 GB. `containers/build.sh <stage> --from-docker docker://...`
pulls a prebuilt image instead. Full walkthrough: [Cluster Setup
Guide](docs/CLUSTER_SETUP.md).

> **Cache paths are host paths.** `esmc.cache_dir` / `esmfold.cache_dir` /
> `openfold.cache_dir` are bound read-only into the container at `/models/hf`
> (and `/models/openfold`) — don't set them to the in-container path.

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

**Every single-point mutant of one sequence** (saturation mutagenesis input):
```bash
bash bash_scripts/generate_satmut.sh --input wt.fasta --output-dir muts/ --dry-run
bash bash_scripts/generate_satmut.sh --input wt.fasta --output-dir muts/
```
Writes `L x 19` FASTAs — 4522 for a 238-residue protein — plus an `index.csv`
mapping each file to its mutation. Run `--dry-run` first to see the count;
`--positions '1-50'` restricts it to a region. Use this when you want a real
embedding or structure *per mutant*: the Saturation Mutagenesis tab scores every
substitution from a single forward pass on the wild type and never embeds the
mutants themselves.

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

A Streamlit front-end covering config editing, the resource estimator, launch,
live monitoring, results, and saturation mutagenesis. On Kempner it is the
shortest path from a clone to a running pipeline.

### Launching

After [Setup](#setup-kempner), on a login node:

```bash
tmux new -s protforge          # so the app survives an SSH disconnect
module load python && mamba activate "$PROTFORGE_ASSETS/envs/host"
cd "$PROTFORGE_ROOT/ProtForge"
streamlit run webapp/app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
```

The app launches Snakemake as a child process, so it inherits this shell's
environment — activate before starting it, not after.

Then from your laptop, tunnel in and open <http://localhost:8501>:

```bash
ssh -L 8501:localhost:8501 <you>@holylogin06.rc.fas.harvard.edu
```

### In the browser

There is **no `config.yaml` to copy**. The first launch creates a *Default*
session already seeded from `config.kempner.template.yaml`, so the shared
MSA/Boltz databases, partitions, container runtime, images and weight caches
arrive filled in. On the **Configuration** tab you supply only:

| Field | Value |
|-------|-------|
| `slurm.account` | your `kempner_<pi>_lab` account |
| `slurm.email` | where SLURM sends job notifications — leave blank to disable them |
| `input.fasta_dir` | a directory of `.fasta` files, one sequence per file |

Then **Run Pipeline → Launch**. The launch is blocked, with a message naming the
specific problem, if the account is unset, either environment variable is
unexported, or an enabled stage's image or weight cache is missing — so a
half-finished setup fails in the browser, not four hours into a queue.

Each session keeps its own config, logs and run state under `.sessions/<id>/`
(never in the repo, and never shared between users). Create more from the sidebar
to keep several configurations side by side.

See the [Web UI guide](docs/WEBAPP.md) for VS Code / Open OnDemand access and
login-node etiquette.

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
├── bash_scripts/                # Data preparation (generate_data.sh, generate_satmut.sh)
├── utils/                       # Python utilities (mutations, file conversion)
├── profiles/slurm/              # Snakemake SLURM executor profile
├── .claude/skills/              # Claude Code skill suite (setup, run-pipeline, …)
└── docs/                        # Documentation
```
