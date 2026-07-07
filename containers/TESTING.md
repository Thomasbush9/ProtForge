# Container testing on Kempner

Step-by-step for validating the built per-stage SIFs on the cluster. Assumes
you've already run `bash containers/build.sh all` (or a subset) successfully —
see `containers/README.md`.

Workspace layout assumed throughout:

```
$PROTFORGE_ROOT/
├── ProtForge/        <- repo (REPO_ROOT)
├── sifs/             <- built SIFs (msa.sif, boltz.sif, esm.sif, openfold.sif)
├── models/hf/        <- mounted ESM-C + ESMFold HF cache
├── models/openfold/  <- OpenFold3 weights + CCD cache
├── sing_cache/       <- singularity layer cache
└── sing_tmp/         <- build staging
```

Set once per shell (substitute your workspace):

```bash
export PROTFORGE_ROOT=/n/holylfs06/LABS/<your_lab>/Everyone/<you>
export SIF_DIR=$PROTFORGE_ROOT/sifs
```

---

## Step 0 — Sanity-check the build artifacts

```bash
ls -lh "$SIF_DIR"/*.sif
# Expect multi-GB per stage. If missing, find anywhere they might have landed:
find "$PROTFORGE_ROOT" -name '*.sif' -exec ls -lh {} \;
```

Identify the runtime (informational — both work):

```bash
singularity --version
# Apptainer     -> "apptainer version X.Y.Z"
# SingularityCE -> "singularity-ce version X.Y.Z"
ls -la "$(which singularity)"   # is it a symlink to apptainer?
```

Inspect an image:

```bash
singularity inspect "$SIF_DIR/boltz.sif"
```

---

## Step 1 — Per-stage smoke tests (model bind only, no DB bind-mounts)

Each stage image has a test script under `containers/test/`. They exercise the
model-cache bind and a tiny end-to-end fold/predict, but not the MSA/Boltz DB
binds. There is no single `smoke.sh` — run the per-stage scripts.

Populate the model cache once before running the ESM/OpenFold tests:

```bash
cd "$PROTFORGE_ROOT/ProtForge"
python scripts/download_models.py --cache-dir "$PROTFORGE_ROOT/models/hf"
```

Get a GPU node first:

```bash
salloc -p kempner_h100 --account=<your_account> --gres=gpu:1 -t 30 --mem=32G
```

Edit the path variables at the top of each script (SIF path, input fixture,
output dir), then run the ones for stages you built:

```bash
cd "$PROTFORGE_ROOT/ProtForge"
bash containers/test/esmfold2_test_image.sh    # ESM image — no DB binds needed
bash containers/test/test_batch_esmc.sh        # ESM-C batch
bash containers/test/openfold3_test_image.sh   # OpenFold image
bash containers/test/msa_test_image.sh         # MSA image — needs shared DB binds
bash containers/test/boltz_test_image.sh       # Boltz image — needs shared DB binds
```

**If one fails**, common causes:
- not on a GPU node, or `--nv` not honored (`nvidia-smi` fails in-container).
- CUDA driver/runtime mismatch (`torch.cuda.is_available()` is False).
- missing pip install in the def's `%post`.
- mounted HF cache missing or not bound to `/models/hf`.

---

## Step 2 — End-to-end (one real protein, full pipeline)

Runs MSA → Boltz → ESM-C → ESMFold on a single ~76 aa protein through the
per-stage SIFs + bind-mounted ColabFold and Boltz DBs. Validates the runtime
path production jobs use, minus the webapp layer.

### Test sequence

Ubiquitin (76 aa, well-conserved, fast MSA). Save as `containers/test/e2e.fasta`:

```
>ubiquitin_test
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
```

### Config

Copy `config.template.yaml` to `containers/test/e2e_config.yaml` and set the
container/SLURM/IO fields (short recycling for speed). The container block uses
the **per-stage** keys — no single `gpu` SIF, no global `bind_paths`:

```yaml
pipeline:
  msa: true
  boltz: true
  esmc: true
  esmfold: true
  openfold: false

input:
  fasta_dir: containers/test/e2e_in
output:
  parent_dir: containers/test/e2e_out

msa:
  max_files_per_job: 1
  array_max_concurrency: 1
  mmseq2_db:    /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db
  colabfold_db: /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/colabfold_db
boltz:
  max_files_per_job: 1
  array_max_concurrency: 1
  recycling_steps: 3       # short test; production uses 10
  diffusion_samples: 5     # short test
  samples_to_save: 1
  num_runs: 1
  cache_dir: /n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db
esmc:
  models: [600M]
  max_files_per_job: 1
  cache_dir: /models/hf
esmfold:
  max_files_per_job: 1
  cache_dir: /models/hf

containers:
  runtime: auto
  colabfold: /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/msa.sif
  boltz:     /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/boltz.sif
  esmc:      /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/esm.sif
  esmfold:   /n/holylfs06/LABS/<your_lab>/Everyone/<you>/sifs/esm.sif

slurm:
  log_dir: containers/test/e2e_out/job_logs
  partition: kempner_requeue
  account: <your_slurm_account>
  email: <you>@example.com
```

Each rule builds its own bind set (inputs + shared DBs read-only + the host
`cache_dir` mapped to `/models/hf`); there is no hand-maintained global bind
list. `esmc.cache_dir` / `esmfold.cache_dir` are the in-container mount point
(`/models/hf`), populated on the host by `download_models.py`.

### Stage inputs and dry-run

```bash
cd "$PROTFORGE_ROOT/ProtForge"
mkdir -p containers/test/e2e_in containers/test/e2e_out
cp containers/test/e2e.fasta containers/test/e2e_in/

snakemake --profile profiles/slurm/ \
    --configfile containers/test/e2e_config.yaml -n --reason
```

Pass criterion: the DAG prints without errors and each rule's shell preview
shows a `singularity exec …` invocation.

### Real run

```bash
snakemake --profile profiles/slurm/ \
    --configfile containers/test/e2e_config.yaml --rerun-incomplete

squeue -u $USER
tail -f containers/test/e2e_out/job_logs/*.out
```

Expected wall time: ~20–40 min depending on queue (MSA dominates against the
700 GB DB).

### Validate outputs

Outputs land under `{parent_dir}/sequences/{seq}/`:

```bash
O=containers/test/e2e_out/sequences/ubiquitin_test
ls -lh "$O"/msa/*.a3m                                   # 1. MSA
find "$O/boltz" -name '*_model_0.cif' -exec ls -lh {} \;   # 2. Boltz structure
ls -lh "$O"/esmc/600M/                                  # 3. ESM-C embeddings
ls -lh "$O"/esmfold/fast/structure.cif                  # 4. ESMFold structure
```

Pass criterion: all four return non-empty, non-zero-size files.

---

## Step 3 — Webapp path (final integration)

Only attempt after Step 2 passes. If the webapp fails but Step 2 passed, the
bug is in config generation / job submission / status tracking — not the
container.

---

## Cleanup

```bash
rm -rf containers/test/e2e_out containers/test/_smoke_out
```

---

## See also

- `containers/README.md` — build / pull / env-var setup
- `docs/CLUSTER_SETUP.md` — full first-time install walkthrough
