# Container testing on Kempner

Step-by-step for validating the built `protforge-gpu.sif` on the cluster.
Assumes you've already run `bash containers/build.sh` successfully (see
`containers/README.md`).

Workspace layout assumed throughout:

```
$TBUSH/container/
├── ProtForge/        <- repo (REPO_ROOT)
├── sifs/             <- built SIFs
├── models/hf/        <- mounted ESM-C + ESMFold HF cache
├── sing_cache/       <- singularity layer cache
└── sing_tmp/         <- build staging (created on demand)
```

Set once per shell:

```bash
export TBUSH=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush
export PROTFORGE_ROOT=$TBUSH/container
export SIF=$PROTFORGE_ROOT/sifs/protforge-gpu.sif
```

---

## Step 0 — Sanity-check the build artifact

Before testing anything, confirm the SIF exists and is non-trivially sized.

```bash
ls -lh "$SIF"
# Expect multi-GB, but smaller than the old fat image with baked weights.
# If missing, find anywhere it might have landed:
find "$TBUSH" -name 'protforge-gpu.sif' -exec ls -lh {} \;
```

Identify the runtime (informational — both work):

```bash
singularity --version
# Apptainer  -> "apptainer version X.Y.Z"
# SingularityCE -> "singularity-ce version X.Y.Z" or "singularity version X.Y.Z"
which singularity
ls -la $(which singularity)   # is it a symlink to apptainer?
```

Inspect the image labels:

```bash
singularity inspect "$SIF"
# Expect Author / Version / Description from %labels in the .def file.
```

---

## Step 1 — Smoke test (model bind only, no DB bind-mounts)

Automated check: GPU visible → PyTorch+CUDA → tools importable → mounted
weights load → ESMFold folds a 49 aa peptide end-to-end. It exercises the
model-cache bind, but not MSA / Boltz DB binds.

Populate the model cache once before running:

```bash
cd "$PROTFORGE_ROOT/ProtForge"
python scripts/download_models.py --cache-dir "$PROTFORGE_ROOT/models/hf"
```

Get a GPU node first:

```bash
salloc -p kempner_h100 --account=<your_account> --gres=gpu:1 -t 30 --mem=32G
```

Run:

```bash
cd "$PROTFORGE_ROOT/ProtForge"
bash containers/test/smoke.sh   # if absent, run the per-stage containers/test/*_test_image.sh
```

Pass criterion: prints `=== ALL SMOKE TESTS PASSED ===` at the end.
Each step prints a `=== [N/7] ... ===` header, so failures are localized.

Typical runtime: ~3-5 min. ESMFold step dominates (~2 min on H100).

**If it fails**, paste the failing `=== [N/7] ===` block. Common causes:
- step 1 (`nvidia-smi`): not on a GPU node, or `--nv` not honored.
- step 2 (`torch.cuda`): CUDA driver/runtime mismatch.
- step 4 (tools): missing pip install in `%post` (e.g., the httpx case from 2026-05-14).
- step 5 (weights): mounted HF cache missing or not bound to `/models/hf`.
- step 6 (ESMFold end-to-end): OOM, bad CUDA, or the model failed to load.

---

## Step 2 — Stage-1 E2E (one real protein, full pipeline)

Runs MSA → Boltz → ESM → ESMFold on a single ~76 aa protein through the SIF
+ bind-mounted ColabFold and Boltz DBs. Validates the runtime path that
production jobs will use, minus the webapp layer.

**Not yet scripted** — manual recipe for now; will be wrapped in
`containers/test/e2e.sh` once smoke passes consistently.

### Bind-mount targets

| Host path | Container path | Mode |
|---|---|---|
| `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/colabfold/databases` | `/data/colabfold_db` | ro |
| `/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db` | `/data/boltz_db` | ro |
| `$PROTFORGE_ROOT/models/hf` | `/models/hf` | ro |
| `$PWD` (working dir) | `$PWD` | rw |

### Test sequence

Ubiquitin (76 aa, well-conserved, fast MSA). Save as `containers/test/e2e.fasta`:

```
>ubiquitin_test
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
```

(Do this once, then `git add` so future runs reuse it.)

### Hand-crafted config

Create `containers/test/e2e_config.yaml` (substitute `<ACCT>` for your SLURM
account):

```yaml
pipeline:
  msa: true
  boltz: true
  esm: true
  esmfold: true
  es: false

input:
  fasta_dir: containers/test/e2e_in

output:
  parent_dir: containers/test/e2e_out

msa:
  max_files_per_job: 1
  array_max_concurrency: 1
  mmseq2_db: /data/colabfold_db
  colabfold_db: /data/colabfold_db
  colabfold_bin: /opt/conda/envs/colabfold/bin  # adjust if path inside SIF differs

boltz:
  max_files_per_job: 1
  array_max_concurrency: 1
  recycling_steps: 3       # short test; production uses 10
  diffusion_samples: 5     # short test
  samples_to_save: 1
  num_runs: 1
  cache_dir: /data/boltz_db
  colabfold_db: /data/colabfold_db

esm:
  num_chunks: 1
  array_max_concurrency: 1
  cache_dir: /models/hf

esmfold:
  input_type: yaml
  num_chunks: 1
  array_max_concurrency: 1
  cache_dir: /models/hf

containers:
  runtime: auto
  gpu: /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/container/sifs/protforge-gpu.sif
  bind_paths: "/n/holylfs06/LABS/kempner_shared/Everyone/workflow/colabfold/databases:/data/colabfold_db:ro,/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db:/data/boltz_db:ro,/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/container/models/hf:/models/hf:ro,/n/holylfs06,/n/home06"

slurm:
  log_dir: containers/test/e2e_out/job_logs
  partition: kempner_requeue
  account: <ACCT>
  email: thomasbush52@gmail.com
```

Notes:
- `containers.gpu` points all GPU stages at the same SIF.
- `containers.bind_paths` uses `host:container:mode` indirection for DBs and
  model caches.
- Tiny `recycling_steps` / `diffusion_samples` are for test speed; restore
  production values for real runs.

### Stage in the test sequence

```bash
cd "$PROTFORGE_ROOT/ProtForge"
mkdir -p containers/test/e2e_in containers/test/e2e_out
cp containers/test/e2e.fasta containers/test/e2e_in/
```

### Dry-run first

From a login or interactive node (no GPU needed for the DAG check):

```bash
snakemake --profile profiles/slurm/ \
    --configfile containers/test/e2e_config.yaml \
    -n --reason
```

Pass criterion: prints the DAG without errors, every rule shows `container:`
or `singularity exec ...` in its shell preview.

### Real run

```bash
snakemake --profile profiles/slurm/ \
    --configfile containers/test/e2e_config.yaml \
    --rerun-incomplete
```

Watch progress:

```bash
squeue -u $USER
tail -f containers/test/e2e_out/job_logs/*.out
```

Expected wall time: ~20-40 min depending on queue. MSA is usually the
slowest (~10 min for a single sequence against a 700 GB DB).

### Validate outputs

```bash
# 1. MSA (A3M)
ls -lh containers/test/e2e_out/msa/ubiquitin_test/*.a3m
wc -l containers/test/e2e_out/msa/ubiquitin_test/*.a3m   # >> 1 line = real MSA

# 2. Boltz structure (CIF)
find containers/test/e2e_out/boltz -name '*_model_0.cif' -exec ls -lh {} \;

# 3. ESM embeddings (.pt)
find containers/test/e2e_out/esm -name 'ubiquitin_test*.pt' -exec ls -lh {} \;

# 4. ESMFold structure (PDB)
find containers/test/e2e_out/esmfold -name 'ubiquitin_test*.pdb' -exec ls -lh {} \;
```

Pass criterion: all four file globs return non-empty, non-zero-size files.

---

## Step 3 — Webapp path (final integration)

Only attempt this after Step 2 passes. If the webapp fails but Step 2
passed, the bug is in config generation / job submission / status tracking —
not the container. That's the bisect property we wanted.

(Webapp testing recipe TBD — depends on which deploy mode you use to point
the webapp at this SIF.)

---

## Cleanup

Test outputs accumulate under `containers/test/e2e_out/` and `_smoke_out/`.
Safe to delete between runs:

```bash
rm -rf containers/test/e2e_out containers/test/_smoke_out
```

---

## See also

- `containers/README.md` — build / pull / env-var setup
- `~/Documents/Vault/Notes/Lab/protforge/container-audit.md` — open audit
  items (15 issues from the 2026-05-16 audit)
- `~/Documents/Vault/Notes/Lab/protforge/log/2026-05-15-build-sh-bugfixes.md` —
  history of what we've already fixed
