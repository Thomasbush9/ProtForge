# ProtForge install guide (Kempner)

From zero to a working pipeline. ProtForge runs each GPU stage inside its own
Singularity/Apptainer image; the big MSA/Boltz databases and the model-weight
caches are bind-mounted at runtime. You only build the images for the stages
you actually run.

> **Easiest path — let Claude Code do it.** The repo ships a `setup` skill
> (`.claude/skills/setup/`). From the repo root run `claude`, then say
> *"set up ProtForge"* — it interviews you for your workspace/SLURM/storage
> choices and runs the steps below (workspace dirs, `config.yaml`, image build,
> weight download, smoke test). The rest of this doc is the same process by
> hand. See the README's "drive ProtForge with Claude Code" section.

---

## 0. Prereqs

```bash
# Kempner account + storage. ProtForge needs tens of GB free on /n/holylfs06
# for the SIFs + model cache + outputs. Home dirs are too small.

# Clone the repo into your workspace (NOT into the SIF output dir — see
# containers/README.md for layout).
WORKSPACE=/n/holylfs06/LABS/<your_lab>/Everyone/<you>
git clone https://github.com/Thomasbush9/ProtForge.git "$WORKSPACE/ProtForge"
cd "$WORKSPACE/ProtForge"

# Snakemake (in your personal python/conda env)
pip install snakemake snakemake-executor-plugin-slurm
```

Copy the config template. **On Kempner, use the Kempner template** — the shared
MSA/Boltz databases, partition, container runtime and image/cache layout are
already filled in, so you only set your account and email:

```bash
export PROTFORGE_ROOT="$WORKSPACE"      # YOUR workspace: inputs, outputs, logs
export PROTFORGE_ASSETS="$WORKSPACE"    # images + weights; see below
cp config.kempner.template.yaml config.yaml
# Edit exactly three things: slurm.account, slurm.email, input.fasta_dir.
```

The template splits paths across **two** variables so a lab can share the heavy,
read-only artifacts:

| Variable | Holds | Share it? |
|----------|-------|-----------|
| `PROTFORGE_ASSETS` | container images (`sifs/`) and model weights (`models/`) | Yes — point several users at one copy and none of them build anything |
| `PROTFORGE_ROOT` | your inputs, outputs and job logs | No — per user |

Set both to your workspace if you are building your own images (steps 1–2 below).
If someone has already published a shared copy, point `PROTFORGE_ASSETS` at it and
**skip steps 1 and 2 entirely** — on Kempner the bsabatini lab keeps one at
`/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge-assets`.

The workflow expands both (and `$VAR` / `~`) from your environment at load time.
Export them in the same shell you run `snakemake` from — if one is unset the run
stops immediately with an error naming the offending config key, rather than
building a broken path. Literal absolute paths still work if you prefer them.

`config.template.yaml` remains the cluster-agnostic, fully annotated reference —
use it if you are not on Kempner or want to see every available parameter.

Every other `config.*.yaml` is git-ignored (it holds your paths/account/email);
only the two templates are tracked. Nothing user-specific ships with a clone, so
a config you did not write can only have come from a workspace you copied.

**Using the webapp instead?** You can skip the `cp` above. On first launch the
webapp creates a Default session seeded from `config.kempner.template.yaml`, with
`${PROTFORGE_ROOT}` expanded from your environment and `slurm.account` /
`slurm.email` left blank for you to fill in on the Configuration tab. If a
`config.yaml` already exists in the repo root it seeds from that instead. Either
way the session config lives in `.sessions/<id>/config.yaml`, never in the repo
root, and the run is blocked before submission while the account is unset.

---

## 1. Build the stage images (one-time)

ProtForge uses one image per GPU stage:

| Stage | Def | SIF | Serves |
|---|---|---|---|
| MSA | `msa.def` | `msa.sif` | `colabfold_search` + `mmseqs2` |
| Boltz | `boltz.def` | `boltz.sif` | Boltz structure prediction |
| ESM | `esmfold_cu.def` | `esm.sif` | ESM-C embeddings + ESMFold2 |
| OpenFold | `openfold.def` | `openfold.sif` | OpenFold3 |

**Skip this whole step** if `PROTFORGE_ASSETS` points at a shared copy someone
has already built — that is the point of the variable.

Otherwise build from an **interactive allocation, not a login node**. SIFs land
in `$PROTFORGE_ASSETS/sifs/`, which is where the configs look for them.

```bash
salloc -p test --account=<your_account> -t 4:00:00 --mem 32G --ntasks-per-node 4

#   $PROTFORGE_ASSETS/        (shareable — read-only once built)
#   ├── sifs/                 <- output SIFs land here
#   ├── models/hf/            <- ESM-C + ESMFold HF cache
#   ├── models/openfold/      <- OpenFold3 weights + CCD cache
#   ├── sing_cache/           <- singularity layer cache
#   └── sing_tmp/             <- build staging
#   $PROTFORGE_ROOT/          (yours alone)
#   ├── ProtForge/            <- repo (cloned in step 0)
#   ├── data/, outputs/, job_logs/
export PROTFORGE_ROOT=/n/holylfs06/LABS/<your_lab>/Everyone/<you>
export PROTFORGE_ASSETS="$PROTFORGE_ROOT"      # or a dir you intend to share
mkdir -p "$PROTFORGE_ASSETS"/{sifs,models/hf,models/openfold}

cd "$PROTFORGE_ROOT/ProtForge"
bash containers/build.sh all              # or a subset: bash containers/build.sh boltz esm
python scripts/download_models.py --cache-dir "$PROTFORGE_ASSETS/models/hf"
```

`build.sh` prints `Done. Image at: …` per stage and writes a `.sha256` sidecar
next to each SIF. If a build bails, fix and re-run — `--force` overwrites
partial SIFs. If `--fakeroot` is not permitted on your partition, build/push to
GHCR elsewhere and pull per stage:
`bash containers/build.sh boltz --from-docker docker://ghcr.io/<owner>/protforge-boltz:latest`.

Full build details: `containers/README.md`.

### Validate

```bash
# On a GPU node:
salloc -p kempner_h100 --account=<your_account> --gres=gpu:1 -t 30 --mem=32G

# Per-stage smoke tests (edit the path vars at the top of each first):
bash containers/test/esmfold2_test_image.sh   # ESM image, no DB binds needed
bash containers/test/boltz_test_image.sh      # Boltz image (needs shared DB binds)
```

End-to-end test recipe: `containers/TESTING.md`.

### Point config.yaml at the SIFs

Under `containers:` in `config.yaml`:

```yaml
containers:
  runtime: auto                       # auto | singularity | apptainer
  colabfold: /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/msa.sif
  boltz:     /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/boltz.sif
  esmc:      /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/esm.sif   # esmc + esmfold share the ESM image
  esmfold:   /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/esm.sif
  openfold:  /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/openfold.sif
  # gpu: ""                           # optional shared fallback if a stage key is empty
```

Each rule binds its own inputs, the shared DBs (read-only), and the model cache
per stage — no global bind list to maintain. The model caches go in the stage
sections, as **host** paths the rule maps into the container:

```yaml
msa:
  mmseq2_db:    /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db   # shared, no changes
  colabfold_db: /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/colabfold_db
boltz:
  cache_dir:    /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db    # shared, no changes
esmc:
  cache_dir:    /n/holylfs06/LABS/<your_lab>/Everyone/<you>/models/hf                # host HF cache → /models/hf
esmfold:
  cache_dir:    /n/holylfs06/LABS/<your_lab>/Everyone/<you>/models/hf
openfold:
  cache_dir:    /n/holylfs06/LABS/<your_lab>/Everyone/<you>/models/openfold
```

---

## Per-stage requirements (derived from `workflow/rules/*.smk`)

If you can satisfy the right column for each stage you care about, the pipeline
will run. In container mode the tools are baked into the stage SIF; the DBs and
model caches are bind-mounted.

| Stage | Tools / binaries | Model weights | DBs |
|---|---|---|---|
| MSA | `colabfold_search`, `mmseqs2` | — | ColabFold DB (~700 GB), MMseqs2 DB |
| Boltz | `boltz` CLI, PyTorch + CUDA | Boltz checkpoint (~5 GB) | — |
| ESM-C | `esm` SDK (NOT `fair-esm`), PyTorch + CUDA | `esmc_600m` (~2.5 GB) | — |
| ESMFold | `transformers≥4.40`, PyTorch + CUDA | `facebook/esmfold_v1` (~8 GB) | — |
| OpenFold | OpenFold3, PyTorch + CUDA | OpenFold3 weights | ColabFold MSA (reuses MSA stage) |

---

## Shared resources (already on Kempner — leave the template defaults)

| What | Path |
|---|---|
| ColabFold DBs | `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/colabfold/databases` |
| MMseqs2 DB | `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db` |
| Boltz checkpoint | `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db` |

---

## SLURM section of `config.yaml`

```yaml
slurm:
  log_dir: /n/holylfs06/LABS/<your_lab>/Everyone/<you>/job_logs
  partition: kempner_requeue
  account: <your_slurm_account>       # e.g. kempner_<pi>_lab
  email: <you>@example.com            # SLURM job notifications
```

---

## First run

```bash
cd "$PROTFORGE_ROOT/ProtForge"
snakemake --profile profiles/slurm/ -n     # dry run: shows the DAG
snakemake --profile profiles/slurm/        # launch (uses SLURM)
# resume after a failure/preemption: just re-run, or:
snakemake --profile profiles/slurm/ --rerun-incomplete
```

Monitor:

```bash
squeue -u $USER
tail -f <log_dir>/*.out
```

---

## Troubleshooting

**`bash containers/build.sh` exits "no stage selected"** — pass a stage:
`bash containers/build.sh all` (or `boltz esm …`). The bare command no longer
builds anything by design.

**`ModuleNotFoundError: No module named 'utils'`** — the rule sets `PYTHONPATH`
to the repo root; if running a helper script manually, `export PYTHONPATH=$(pwd)`.

**Container `--cleanenv` strips a needed env var** — rules forward host vars
explicitly (`{container_cmd} --env FOO=$FOO …`, see `boltz.smk`). If you hit
this in a rule, add the `--env` flag inside that rule rather than dropping
`--cleanenv`.

**Empty `sifs/` after a "successful" build** — locate the SIF:
`find "$PROTFORGE_ROOT" -name '*.sif' -exec ls -lh {} \;` — most likely you're
`ls`-ing a stale empty dir at the wrong level.

---

## See also

- `containers/README.md` — per-stage container build details, layout, weights
- `containers/TESTING.md` — smoke + end-to-end test recipes on the cluster
- `docs/SNAKEMAKE_GUIDE.md` — workflow internals (rules, chunking, retries)
- `docs/WEBAPP.md` — Streamlit UI access (SSH / VS Code / Open OnDemand)
