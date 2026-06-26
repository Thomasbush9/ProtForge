# ProtForge install guide (Kempner)

From zero to a working pipeline. Two paths — pick one — both run the same
Snakemake DAG.

| Path | What you get | When to pick |
|---|---|---|
| **A. Container (single SIF)** | One `.sif` with all GPU-stage tools. Bind-mount the big DBs and ESM/ESMFold model cache at runtime. | Reproducible, shareable, no env churn. **Currently in late beta — works but `containers/TESTING.md` Step 2 hasn't been validated end-to-end yet.** |
| **B. Conda / `module load`** | Shared conda envs + downloaded weights + Kempner `module load` for CUDA. The legacy path. | What's known to work today. Fall back here if path A breaks. |

You can mix-and-match per stage (`containers.boltz: /path/to.sif`, leave
`containers.esm: ""` to keep ESM on conda). See "Per-stage requirements"
below to confirm what each stage needs in either mode.

---

## 0. Prereqs (both paths)

```bash
# Kempner account + storage. ProtForge needs ~20 GB free on /n/holylfs06
# for the SIF + cache + outputs. Home dirs are too small.

# Clone the repo into your workspace (NOT into the SIF output dir — see
# containers/README.md for layout)
WORKSPACE=/n/holylfs06/LABS/<your_lab>/Everyone/<you>
git clone https://github.com/Thomasbush9/ProtForge.git "$WORKSPACE/ProtForge"
cd "$WORKSPACE/ProtForge"

# Snakemake (in your personal python env)
pip install snakemake snakemake-executor-plugin-slurm
```

Copy the config template:

```bash
cp config.template.yaml config.yaml
# Edit config.yaml — see "Filling in config.yaml" below.
```

---

## Path A — Container

### A.1 Build the SIF (one-time, ~30 min on a compute node)

Detailed instructions in `containers/README.md`. Summary:

```bash
# On an interactive node — NOT a login node:
salloc -p test --account=<your_account> -t 4:00:00 --mem 32G --ntasks-per-node 4

# Workspace layout: $PROTFORGE_ROOT is the *parent* of the repo, with
# sifs/, model cache, and Singularity cache/tmp dirs as siblings:
#   $PROTFORGE_ROOT/
#   ├── ProtForge/        <- repo (cloned in step 0)
#   ├── sifs/             <- output SIFs land here
#   ├── models/hf/        <- ESM-C + ESMFold HF cache
#   ├── sing_cache/       <- singularity layer cache
#   └── sing_tmp/         <- build staging
export PROTFORGE_ROOT=/n/holylfs06/LABS/<your_lab>/Everyone/<you>
mkdir -p "$PROTFORGE_ROOT/sifs" "$PROTFORGE_ROOT/models/hf"

cd "$PROTFORGE_ROOT/ProtForge"
bash containers/build.sh                  # writes $PROTFORGE_ROOT/sifs/protforge-gpu.sif
python scripts/download_models.py --cache-dir "$PROTFORGE_ROOT/models/hf"
```

The script prints `Runtime : ...` and `Done. Image at: ...` on success.
If it bails, fix and re-run — `--force` overwrites partial SIFs.

### A.2 Validate

```bash
# Smoke test (5 min, needs a GPU node)
salloc -p kempner_h100 --account=<your_account> --gres=gpu:1 -t 30 --mem=32G
bash containers/test/smoke.sh   # if absent, run the per-stage containers/test/*_test_image.sh
# Expect: "=== ALL SMOKE TESTS PASSED ==="
```

End-to-end test recipe in `containers/TESTING.md`.

### A.3 Tell `config.yaml` to use the SIF

Under `containers:` in `config.yaml`, set `containers.gpu` to the SIF:

```yaml
containers:
  runtime: auto             # auto | singularity | apptainer
  gpu: /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/protforge-gpu.sif
  pdanalysis: ""            # no MPI image yet; ES stage stays on conda
  bind_paths: "/n/holylfs06/LABS/kempner_shared/Everyone/workflow/colabfold/databases:/data/colabfold_db:ro,/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db:/data/boltz_db:ro,/n/holylfs06/LABS/<your_lab>/Everyone/<you>/models/hf:/models/hf:ro,/n/holylfs06,/n/home06"
```

The `host:container:ro` syntax binds the shared DBs read-only at
predictable in-container paths. The trailing `/n/holylfs06,/n/home06`
entries are blanket binds for the workspace and user homes — to be
slimmed (audit item H3).

When using the container path, set the in-container paths in the stage
sections:

```yaml
msa:
  mmseq2_db:    /data/colabfold_db   # bind-mounted from host above
  colabfold_db: /data/colabfold_db
boltz:
  cache_dir:    /data/boltz_db
esm:
  cache_dir:    /models/hf
esmfold:
  cache_dir:    /models/hf
```

---

## Path B — Conda / module load (legacy, working today)

### B.1 Automated setup

```bash
cd "$WORKSPACE/ProtForge"
bash setup.sh
```

Prompts for shared base dir, SLURM account, email, FASTA dir, output dir.
Creates shared envs (`esm`, `es-analysis`), downloads ESM weights, patches
ESM hardcoded paths, generates `config.yaml`.

### B.2 Manual setup (if setup.sh isn't enough)

| Stage | What you need | How to get it |
|---|---|---|
| MSA | `colabfold_search`, `mmseqs2`, MSA DBs | All shared on Kempner — see "Shared resources" table below. No install needed. |
| Boltz | `boltz` CLI, Boltz weights | Shared conda env + `boltz_db` on Kempner. No install. |
| ESM | `esm` SDK, `esmc_600m` weights | `setup.sh` creates the env at `{shared_base}/envs/esm` and downloads weights. Then patch `esm/utils/constants/esm3.py` to point at the shared cache (see Troubleshooting). |
| ESMFold | `transformers>=4.40`, `facebook/esmfold_v1` weights | `python scripts/download_models.py --cache-dir <cache>` populates the HF cache. Create env yourself: `conda create -p {shared_base}/envs/esmfold python=3.12 && pip install transformers accelerate torch`. |
| ES | `PDAnalysis`, `MDAnalysis` | `setup.sh` clones PDAnalysis + creates env at `{shared_base}/envs/es-analysis`. |

### B.3 `config.yaml` for path B

Leave the `containers:` block fields empty (or omit the block). Set
`*.env_path` and `*.cache_dir` to your shared paths. Reference values:

```yaml
msa:
  mmseq2_db:    /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db
  colabfold_db: /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/colabfold_db
  colabfold_bin: /n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz/localcolabfold/colabfold-conda/bin
boltz:
  cache_dir: /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db
  env_path:  /n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz
esm:
  env_path:  {shared_base}/envs/esm
  cache_dir: {shared_base}/esm_models_cache
esmfold:
  env_path:  {shared_base}/envs/esmfold
  cache_dir: {shared_base}/esm_models_cache
es:
  env_path:     {shared_base}/envs/es-analysis
  pdanalysis_dir: {shared_base}/PDAnalysis
```

---

## Per-stage requirements (derived from `workflow/rules/*.smk`)

This table is the source of truth. If you can satisfy the right column for
each stage you care about, the pipeline will run.

| Stage | Tools / binaries | Model weights | DBs |
|---|---|---|---|
| MSA | `colabfold_search`, `mmseqs2` | — | ColabFold DB (~700 GB), MMseqs2 DB |
| Boltz | `boltz` CLI, PyTorch + CUDA | Boltz checkpoint (~5 GB) | — |
| ESM | `esm` SDK (NOT `fair-esm`), PyTorch + CUDA | `esmc_600m` (~2.5 GB) | — |
| ESMFold | `transformers≥4.40`, PyTorch + CUDA | `facebook/esmfold_v1` (~8 GB) | — |
| ES | PDAnalysis, MDAnalysis, MPI | — | — |

Container path: tools are baked into the SIF; DBs and model caches are
bind-mounted. Conda path: install tools into per-stage envs, download weights
to a shared cache, DBs are read directly from the shared paths.

---

## Shared resources (already on Kempner — both paths)

| What | Path |
|---|---|
| ColabFold DBs | `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/colabfold/databases` |
| MMseqs2 DB | `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db` |
| Boltz checkpoint | `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db` |
| Boltz conda env | `/n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz` |
| Local ColabFold bin | `…/boltz/localcolabfold/colabfold-conda/bin` |

---

## SLURM section of `config.yaml` (both paths)

```yaml
slurm:
  log_dir: /n/holylfs06/LABS/<your_lab>/Everyone/<you>/job_logs
  partition: kempner_requeue
  account: <your_slurm_account>
  email: <you>@example.com
```

---

## First run

```bash
cd "$WORKSPACE/ProtForge"
snakemake --profile profiles/slurm/ -n     # dry run: shows the DAG
snakemake --profile profiles/slurm/        # launch (uses SLURM)
# resume after a failure: just re-run, or:
snakemake --profile profiles/slurm/ --rerun-incomplete
```

Monitor:

```bash
squeue -u $USER
tail -f <log_dir>/*.out
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'utils'`** (legacy path) — the rule
sets `PYTHONPATH` to repo root; if running scripts manually,
`export PYTHONPATH=$(pwd)`.

**ESM `FileNotFoundError: .../esmc_600m_2024_12_v0.pth`** — ESM library
hardcodes weight paths in `esm/utils/constants/esm3.py`. Re-run `setup.sh`
or patch manually to point at your `esm.cache_dir`.

**Container `cp: cannot copy a directory into itself`** — `PROTFORGE_ROOT`
is set to the repo path. It must be the *parent* of the repo. See
`containers/README.md`.

**Empty `sifs/` after a "successful" build** — locate the SIF:
`find "$PROTFORGE_ROOT" -name 'protforge-gpu.sif' -exec ls -lh {} \;` —
most likely you're `ls`-ing a stale empty dir at the wrong level.

**Container `--cleanenv` strips a needed env var** — rules should forward
host vars explicitly: `{container_cmd} --env FOO=$FOO ...` (see
`boltz.smk:156`). If you hit this in a rule we haven't migrated, the fix
is to add the `--env` flag inside that rule, not to drop `--cleanenv`.

**Container schema confusion (five `containers.*` fields, one SIF)** — set
all four GPU fields to the same SIF for now. Single-`containers.gpu` field
is a pending refactor (audit H4).

---

## See also

- `containers/README.md` — container build details, env-var setup, sandbox iteration workflow
- `containers/TESTING.md` — smoke test + end-to-end test recipes on the cluster
- `docs/SNAKEMAKE_GUIDE.md` — workflow internals (rules, chunking, retries)
- `~/Documents/Vault/Notes/Lab/protforge/container-audit.md` — open hardening items for the container path
- `docs/CONTAINERS.md` — **superseded.** The original per-stage container design predates the single-SIF decision (2026-05-14). Kept for history; do not follow.
