# Chunking and resource optimization

Turns jobs that already ran into a per-stage cost model, then uses that model to
answer the question a fixed `max_files_per_job` cannot: **how many jobs should
this workload be split into, and what should each one request?**

```
dataset.py  →  observations.csv   one row per SLURM job: workload → requested → used
model.py    →  cost_model.yaml    overhead + per-seq(L), VRAM(L), host RAM(N)
plan.py     →  the recommendation cost/makespan frontier + per-job resources
plots.py    →  validation, scaling, and trade-off figures
```

## The idea in one paragraph

Every job pays a fixed `overhead` — container start, model weights off Lustre,
CUDA init — whether it folds one sequence or five hundred. So total GPU time is
`n_chunks · overhead + work`, while wall-clock is roughly
`per-chunk time · ceil(n_chunks / max_concurrent)`. Splitting finer buys
wall-clock and costs GPU-hours. Where a stage sits on that trade-off depends
entirely on how its overhead compares to its real work, which is why the answer
differs per stage rather than being one global chunk size.

## Running it

```bash
module load python/3.12.8-fasrc01 && conda activate snakemake

# 1. harvest jobs (repeat --run per run; jobstats results are cached)
python scripts/calibrate/optimize/dataset.py \
    --run <OUTPUT_DIR>:<.snakemake/log/....snakemake.log> \
    --out observations.csv

# 2. fit (stair sweeps are optional but supply the length curve)
python scripts/calibrate/optimize/model.py --obs observations.csv \
    --stair <calib_dir>/results.csv --out cost_model.yaml

# 3. plan a workload
python scripts/calibrate/optimize/plan.py --model cost_model.yaml \
    --stage esmfold --fasta-dir <fasta_dir> --max-concurrent 10

# 4. figures (needs matplotlib — use the protein_interpret env)
python scripts/calibrate/optimize/plots.py --obs observations.csv \
    --model cost_model.yaml --stair <calib_dir>/results.csv --out-dir figures/
```

## What the model measured (H100, mid-2026)

| stage | overhead/job | per-seq cost | verdict |
|---|---|---|---|
| msa | ~1670 s *(not truly fixed — see below)* | large, and I/O-bound | **leave at 25/job** — consolidating was tested and is worse |
| esmc | ~65 s | ~0.2 s | **one job** — splitting cost 30x for no speedup |
| openfold | ~90 s | ~11 s | few jobs |
| esmfold | ~449 s | grows with L² | split freely — overhead is ~1% of the work |
| boltz | ~0 s | ~50 s at L≈290 | split freely — splitting is free |

### MSA: where this model failed, and why

The model fit MSA's ~1670 s as fixed per-job overhead and therefore recommended
consolidating 110 sequences into fewer, bigger jobs. **That was tested on
2026-07-18 and it is wrong.** 2 jobs x 55 against the 5 x 25 baseline, same
partition, same 110 sequences:

| | 5 x 25 (baseline) | 2 x 55 (predicted better) |
|---|---|---|
| total GPU-time | 145.9 min | 149.1 min (+2%) |
| makespan | 34.5 min | **74.6 min (+116%)** |
| predicted per job | — | 28–42 min vs **74.6 actual** |

Consolidating was strictly worse on both axes. Two causes, both invisible to a
model whose only inputs are chunk size and sequence length:

1. **That 1670 s is not a fixed startup cost.** Unlike a model load, it is
   colabfold database-scan I/O that partly scales with the query set, so making
   chunks bigger does not amortize it.
2. **MSA is I/O-bound on a shared database** (CPU ~14%, GPU ~5%), so two MSA jobs
   packed onto one node contend for the very resource that sets their runtime —
   measured 1.19x slower at N=25, and SLURM put both N=55 jobs on one node.

`dataset.py` now records `node` and `jobs_on_node` so placement can be modelled
instead of being absorbed as noise. The general lesson: this cost model is
trustworthy for **compute-bound** stages (boltz validates at 1% error) and
should be treated as a hypothesis to test for **I/O-bound** ones.

`overhead` for esmfold is corroborated twice: the stair sweep's
wall-minus-inference gap (448 s) and five production jobs that loaded the model
and folded nothing (449 s).

## Reading the figures

* **validation.png** first. Predicted vs observed wall time per stage. boltz,
  openfold and msa land within 1–5% of the diagonal. esmfold shows a horizontal
  band around 3800 s — a degraded run whose wall time stopped tracking workload
  at all; the fitter rejects those points automatically.
* **scaling.png** — measured VRAM and per-sequence time against length, fits
  overlaid.
* **tradeoff.png** — the decision picture. A flat red curve means splitting is
  free (boltz, esmfold); a steep one means splitting is pure waste (msa, esmc).

## Things worth knowing before trusting a number

* **jobstats is the only GPU source here** (`sacct`'s `gres/gpumem` is always 0
  on this cluster) but it **undersamples host RAM** — it read 33 GB for MSA jobs
  whose real peak was 142 GB. Host memory therefore comes from the snakemake
  benchmark's `max_rss`, and jobs shorter than ~2 min have no GPU data at all.
* **Overhead is only identifiable if N varies.** A run that used one chunk size
  for every job cannot separate "300 s of startup" from "6 s/sequence × 50". The
  fitter detects this and takes overhead from direct evidence instead, but the
  cleanest fix is to vary `max_files_per_job` across runs.
* **Extrapolation is flagged, not hidden.** Recommending 658 sequences per job
  from data that only ever saw 50 is a 13x extrapolation, and `plan.py` says so.
  Verify with one job before launching a fleet.
* **Sequences too long for any GPU are a filtering problem, not a chunking one.**
  A chunk's VRAM is set by its longest member, so one 8000-residue protein
  stalls every chunk it joins. `plan.py` reports these and plans around them.
* **msa's per-sequence cost fit to zero** because production only spanned N=10–25,
  where wall time did not move. That matches the physics (one database scan per
  chunk) but it will not hold forever — do not read it as "sequences are free"
  at N in the thousands.
