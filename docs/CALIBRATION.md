# Resource Calibration

This guide explains how to run a real cluster sweep on a small sample of your
inputs to measure how much memory, GPU time, and CPU each pipeline stage
actually consumes — and how that data later refines the webapp's resource
estimator.

The end goal: replace the heuristic coefficients in
`webapp/scaling_models.yaml` with empirical ones in
`webapp/scaling_models.calibrated.yaml` per `(stage, gpu_type)`.

## When to run this

- **Before** processing a large input batch you've never run at this scale
  (e.g. 7k FASTAs of varied length). The estimator's heuristic numbers are
  ballpark; calibration tightens them.
- **After** changing a Boltz parameter that affects compute (`recycling_steps`,
  `diffusion_samples`, `num_runs`).
- **When adding a new GPU type** (so the estimator can route to it correctly).

You do **not** need to recalibrate for routine runs — the calibrated
coefficients persist in `scaling_models.calibrated.yaml` until you regenerate
them.

## High-level flow

```
            (one-time, login node)
1. Subsample          tests/calibration_inputs/fastas/  (≈20 stratified FASTAs)
                                |
                                v
2. Sweep on cluster   /n/holylfs06/.../calib_<ts>/run/
   (calibrate.sh)        ├── benchmarks/<stage>/*.tsv     (wall time, max RSS)
                         ├── <stage>_chunks/chunk_stats.tsv  (length stats)
                         └── sequences/<seq>/...
                                |
                                v
3. Analyze            Join chunk_stats × benchmarks on chunk_id
                                |
                                v
4. Refit              webapp/scaling_models.calibrated.yaml
   (recalibrate_from_benchmarks)
```

Today, **steps 1–3 are wired**; step 4 (the regression refit) is the next
piece — once we've seen the shape of the data, we'll plug it into
`webapp/estimator.py::recalibrate_from_benchmarks`.

## Step 1 — Subsample your inputs

Pick ~20 FASTAs spanning the length distribution of the real workload.
Random sampling under-represents the long tail, where O(L²) memory
diverges. The subsampler stratifies by length quantile with extra weight
on the upper tail.

```bash
python scripts/calibrate/subsample.py \
    --input_dir /path/to/your/7k_fastas \
    --output_dir tests/calibration_inputs/fastas \
    --n 20 \
    --seed 42
```

What lands in `output_dir`:

```
tests/calibration_inputs/fastas/
  <picked_filename_1>.fasta
  <picked_filename_2>.fasta
  ...
  manifest.csv          filename, length, bin (e.g. q75-q90), source_path
```

Inspect the spread before running the sweep:

```bash
column -t -s, tests/calibration_inputs/fastas/manifest.csv | sort -k2 -n
```

If you already know a particular sequence will OOM on your target GPU
(e.g. >2000 residues on a 40 GB card), drop it from the dir before the
sweep — calibration is for the *working range*, not the failure regime.

## Step 2 — Run the sweep

The pipeline runs from a **login node**, not an interactive `salloc` session.
The Snakemake driver only does DAG scheduling and `sbatch` submissions; the
actual GPU work lands on compute nodes.

```bash
mamba activate snakemake          # per docs/SNAKEMAKE_GUIDE.md
tmux new -s calib                 # so SSH disconnect doesn't kill the driver

bash slurm_scripts/calibrate.sh all h100 \
    tests/calibration_inputs/fastas \
    /n/holylfs06/LABS/bsabatini_lab/Everyone/$USER/calib_$(date +%Y%m%d_%H%M%S)

# Detach: Ctrl-b d
# Reattach: tmux attach -t calib
```

### What `all` mode does

`all` enables `msa + boltz + esm + esmfold` in one Snakemake invocation.
The DAG sequences them automatically via `.msa_complete` / `.boltz_complete`
sentinels — Boltz can't start before all MSA jobs finish, ESM/ESMFold can't
start before Boltz. ES is excluded (CPU-only, calibrate separately if needed).

### Concurrency cap

`CALIB_MAX_JOBS=4` (default) caps simultaneous SLURM jobs so calibration
doesn't crowd the queue. Override:

```bash
CALIB_MAX_JOBS=8 bash slurm_scripts/calibrate.sh all h100 ...
```

With cap=4 and 19 sequences:
- MSA wave: ~5 batches of 4
- Boltz wave: ~5 batches of 4 (slowest = your longest FASTA × `recycling × samples`)
- ESM + ESMFold wave: ~10 batches of 4 (cheap relative to Boltz)

Wall-clock is bounded by Boltz, typically 2–4 hours overnight.

### Other env vars

| Var | Default | Purpose |
|---|---|---|
| `SLURM_ACCOUNT` | `kempner_bsabatini_lab` | Account that has access to your chosen partition. Kempner GPU partitions need the `kempner_*` account, not the lab's plain account. |
| `PROTFORGE_LOG_DIR` | `/n/home06/$USER/job_logs` | Where SLURM job logs go. |
| `PROTFORGE_ESM_ENV` | `/n/home06/$USER/envs/esm` | ESM conda env. |
| `PROTFORGE_ESMFOLD_ENV` | `/n/home06/$USER/envs/esmfold` | ESMFold conda env. |

The shared MSA/Boltz paths (mmseq2_db, colabfold_db, boltz cache, etc.) are
hardcoded in `calibrate.sh` from `config.template.yaml` and don't need
overriding for Kempner users.

### Why `max_files_per_job: 1` for calibration

Production runs use `max_files_per_job: 25` — each SLURM job processes 25
sequences sequentially, and you get one `benchmarks/*.tsv` row per *job*
(aggregating all 25). For calibration we want one row per *length value* so
we can fit `runtime ~ L²` per sequence. So calibrate.sh forces:

- `msa.max_files_per_job: 1`
- `boltz.max_files_per_job: 1`
- `esm.num_chunks: 100`     — capped at file count → 1 sequence per chunk
- `esmfold.num_chunks: 100` — same

**Tradeoff:** each job re-pays its setup overhead (model load, env activation).
For Boltz (~45 min/seq) that's negligible. For ESM (~3 s/seq + ~30 s startup),
calibrated wall times are slightly overestimated vs production where the
startup amortizes across 25 sequences. For sizing SLURM resources with safety
margins that's fine — over-estimating is the safe direction.

If you ever want production-mode aggregate timings, run a separate sweep with
`max_files_per_job: 25` and join against `chunk_stats.tsv` (which records
per-chunk mean/p95/max length).

## Step 3 — What you get back

After the run completes, `$OUT_ROOT/run/` contains:

```
$OUT_ROOT/
├── config.yaml                             # the rendered calibration config
├── logs/                                   # slurm.log_dir
├── summary.txt                             # one-pager: per-stage TSV listing
└── run/
    ├── benchmarks/
    │   ├── msa/colabfold_search_<chunk_id>.tsv
    │   ├── boltz/predict_<chunk_id>_run_<run_id>.tsv
    │   ├── esm/esm_chunk_<chunk_id>.tsv
    │   └── esmfold/<rule>_<chunk_id>.tsv
    ├── msa_chunks/chunk_stats.tsv          # chunk_id, num_seqs, mean_len, ...
    ├── boltz_chunks/chunk_stats.tsv
    ├── esm_chunks/chunk_stats.tsv
    ├── sequences/<seq>/{<seq>.yaml, msa/, boltz/, esm*, esmfold/}
    └── .{msa,boltz,esm,esmfold}_complete
```

### The benchmark TSV schema

Snakemake's `benchmark:` directive writes:

```
s    h:m:s    max_rss    max_vms    max_uss    max_pss    io_in    io_out    mean_load    cpu_time
```

Key columns:
- `s` — wall-clock seconds
- `max_rss` — peak resident memory (MB)
- `cpu_time` — CPU seconds (sum across cores)

### chunk_stats.tsv schema

```
chunk_id    num_seqs    mean_len    min_len    p95_len    max_len    total_residues
```

### Joining the two

The `chunk_id` column is the join key. Example pandas snippet (run from a
notebook on the login node):

```python
import pandas as pd
from pathlib import Path

CALIB = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/$USER/calib_<ts>/run")
STAGE = "boltz"

stats = pd.read_csv(CALIB / f"{STAGE}_chunks/chunk_stats.tsv", sep="\t")
benches = []
for tsv in (CALIB / f"benchmarks/{STAGE}").glob("*.tsv"):
    df = pd.read_csv(tsv, sep="\t")
    # rule output names are like predict_<chunk_id>_run_0
    chunk_id = int(tsv.stem.split("_")[1])
    df["chunk_id"] = chunk_id
    benches.append(df)
bench = pd.concat(benches)

joined = stats.merge(bench, on="chunk_id")
# Now: joined[["mean_len", "p95_len", "s", "max_rss"]] is plottable
```

Plot `s` (wall time) and `max_rss` against `p95_len` (or `mean_len`, since
chunks have one sequence each). For Boltz on H100 you should see roughly
linear runtime up to ~600 residues then quadratic; memory should be
quadratic throughout.

## Step 4 — Refit coefficients (next phase)

Once you have data and have eyeballed the shape, the next change wires
`webapp/estimator.py::recalibrate_from_benchmarks` to:

1. Walk `<calib_dir>/run/benchmarks/<stage>/*.tsv` and the matching
   `<stage>_chunks/chunk_stats.tsv`.
2. Fit `runtime ~ p95_len + p95_len²` and `mem ~ p95_len + p95_len²` per
   `(stage, gpu_type)` via `sklearn.LinearRegression`.
3. Write the fitted coefficients to `webapp/scaling_models.calibrated.yaml`
   in the same schema as `webapp/scaling_models.yaml`.
4. The estimator already prefers the `.calibrated.yaml` over the heuristic
   defaults when the file exists.

That step is gated on having real data — design after the first sweep
lands so we know the shape we're fitting.

## Troubleshooting

### `Invalid account or account/partition combination`

Your `$SLURM_ACCOUNT` doesn't have access to the chosen partition. Check
which accounts you can submit under:

```bash
sacctmgr -nP show assoc user=$USER format=account,partition | sort -u
```

Then re-run with the right one:

```bash
SLURM_ACCOUNT=kempner_<your_lab> bash slurm_scripts/calibrate.sh all h100 ...
```

### `output_dir` on `/tmp` doesn't work

`/tmp` on a login node is **not** visible from compute nodes. Always pass an
explicit `output_dir` on a shared filesystem (`/n/holylfs06/...` or
`/n/home06/...`) so the SLURM jobs can read/write it.

### A single sequence OOMs

Drop it from `tests/calibration_inputs/fastas/` and rerun — calibration is
for the working range, not the failure regime. The estimator's job is to
warn before submission, not to model the OOM threshold itself.

### Resuming after a crash

Snakemake's resume just works:

```bash
bash slurm_scripts/calibrate.sh all h100 <fasta_dir> <existing_output_dir>
```

It reuses the existing `config.yaml` in `<existing_output_dir>` if present
(actually it overwrites it — but with the same content given the same args)
and re-runs only the rules whose outputs are missing. See
`docs/SNAKEMAKE_GUIDE.md` for the underlying behaviour.

## Cross-references

- `docs/SNAKEMAKE_GUIDE.md` — how the Snakemake/SLURM machinery works
- `docs/CLUSTER_SETUP.md` — Kempner-specific setup and shared paths
- `slurm_scripts/calibrate.sh` — the calibration entry point
- `scripts/calibrate/subsample.py` — stratified FASTA picker
- `webapp/scaling_models.yaml` — heuristic coefficients (current source of truth for the estimator)
- `webapp/estimator.py` — pure-Python module that turns input stats into per-stage estimates
