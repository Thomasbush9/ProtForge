# Stair calibration

Per-stage GPU scaling measurement for the ProtForge pipeline: sweep sequence
length, record **GPU memory** and **inference time** for each stage, and turn the
results into the scaling coefficients that drive chunk sizing.

```
pick_rungs.py  →  chunk_fastas.py  →  bench_one.sh (per stage × rung)
               →  collect.py       →  plot_stage_scaling.py  (the L → mem/time plot)
                                    →  fit_and_plot.py        (scaling_models.yaml)
```

The measurement (`bench_one.sh`) runs **inside the GPU SIF under SLURM**; the
setup, collection, plotting, and fitting are CPU-only and run on a login node.

## Environment

The CPU-side tooling needs `pyyaml`, `numpy`, `matplotlib`:

```bash
python -m venv ~/envs/protforge-calibrate
source ~/envs/protforge-calibrate/bin/activate
pip install -r scripts/calibrate/requirements-calibrate.txt
```

(`pyyaml` + `numpy` already ship in `requirements-host.txt`; `matplotlib` is the
only addition. The file lists all three so the env stands alone.)

## 1. Smoke test (no GPU, no cluster)

Before spending GPU hours, prove the whole chain is wired. `stair_test.sh` builds
a tiny synthetic fixture and runs every step with `bench_one.sh --mock`
(length-dependent fake measurements, including a faked OOM to exercise the
drop path), then asserts every artifact exists.

**On the login node** (seconds):

```bash
bash scripts/calibrate/stair/stair_test.sh
# keep the artifacts to inspect:
bash scripts/calibrate/stair/stair_test.sh --work-dir ./stair_test_out --keep
```

**As a SLURM job** (CPU partition — no GPU requested) via `stair_test.slrm`.
Tell it which env to activate with `--export`:

```bash
# venv:
sbatch --partition=<cpu_part> --account=<acct> --time=10 \
       --export=ALL,CALIB_VENV=$HOME/envs/protforge-calibrate \
       scripts/calibrate/stair/stair_test.slrm

# or conda:
sbatch --partition=<cpu_part> --account=<acct> --time=10 \
       --export=ALL,CALIB_CONDA=protforge-calibrate \
       scripts/calibrate/stair/stair_test.slrm
```

Optional `OUT_DIR=/path` in `--export` chooses where artifacts are kept
(default `$SLURM_SUBMIT_DIR/stair_test_out`). The launcher fails early with an
install hint if the env lacks the three packages.

A PASS means the runner↔harness argument contracts, the `results.csv` schema,
`collect.py`'s sort/OOM logic, and both plotters all line up. (It validates
**plumbing, not performance** — the numbers are synthetic.)

## 2. Real sweep (GPU, SLURM)

`run_stage.sh` runs on the **login node** and submits one `sbatch --array` per
stage (one task per rung). Do NOT `sbatch run_stage.sh` itself — it is the
submitter.

```bash
bash scripts/calibrate/stair/run_stage.sh \
    --run-dir calib_h100 --config config.yaml \
    --input-dir <fasta_dir> --gpu-type h100 \
    --min 200 --max 3000 --step 200
# whole node per task for contention-free wall time:
#   --exclusive
# validate setup only (build staircase + chunks, no submit):
#   --setup-only
# print the sbatch commands without submitting:
#   --dry-run
```

Partition / account / log dir / per-stage runtime come from `--config`
(`slurm.*`, `slurm.resources.<stage>.runtime`). Model stages depend on the MSA
array (`afterok`). Results accumulate (flock-appended) in
`calib_h100/results.csv`.

## 3. Plots and scaling coefficients

Once `results.csv` is populated:

```bash
# the length → (GPU memory, inference time) figure, one series per stage:
python scripts/calibrate/plot_stage_scaling.py --run-dir calib_h100
#   → calib_h100/stage_scaling.png

# fit coefficients + a per-stage fit-overlay diagnostic:
python scripts/calibrate/fit_and_plot.py --run-dir calib_h100 --output-dir fits/
#   → fits/scaling_models_<gpu>.yaml   (merge into webapp/scaling_models.yaml)
#   → fits/calibration_fit.png
#   → fits/per_stage_summary.txt
```

`fit_and_plot.py` fits **GPU memory** (`vram_peak_mib`) and **inference time**
(`infer_s`) vs length, status==ok rows only. `esmc_300M/600M/6B` collapse into a
single size-agnostic `esmc` block (largest size by default; `--esmc-from` to
pick another). `collect.py` (run by the sweep) sorts `results.csv` and prints a
per-stage table with the OOM boundary.

## Files

| File | Role |
|---|---|
| `stair/pick_rungs.py` | choose real FASTAs at a length ladder |
| `stair/bench_one.sh` | measure ONE (stage, rung); `--mock` for the test |
| `stair/stair_bench.slrm` | SLURM array wrapper (one task = one rung) |
| `stair/run_stage.sh` | login-node driver; submits the per-stage arrays |
| `stair/collect.py` | sort `results.csv` + per-stage summary / OOM boundary |
| `stair/stair_test.sh` | end-to-end smoke test (no GPU/SLURM) |
| `stair/stair_test.slrm` | run the smoke test as a CPU SLURM job |
| `plot_stage_scaling.py` | length → (GPU mem, time) combined figure |
| `fit_and_plot.py` | fit scaling coefficients → `scaling_models.yaml` |
| `requirements-calibrate.txt` | CPU-side env (pyyaml, numpy, matplotlib) |
