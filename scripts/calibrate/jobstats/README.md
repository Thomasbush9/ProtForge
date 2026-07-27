# Jobstats right-sizing

The `stair/` sweep *deliberately* measures scaling and costs GPU hours. This is
the cheap complement: every production run already left behind SLURM jobs, and
`jobstats` knows what they actually used. Harvest that, compare it to what was
requested, and get a per-stage "ask this next time".

```bash
module load python/3.12.8-fasrc01 && conda activate snakemake

# newest snakemake run in .snakemake/log/, no benchmark cross-check:
python scripts/calibrate/jobstats/harvest.py

# a specific run, with benchmark TSVs folded in (recommended):
python scripts/calibrate/jobstats/harvest.py \
    --log .snakemake/log/<run>.snakemake.log \
    --output-dir <output.parent_dir> \
    --json report.json
```

It maps each SLURM job to a pipeline stage from the snakemake log
(`rule_<name>/…/<jobid>.log`), pulls `jobstats -j`, and prints observed
host-RAM / VRAM / SM-utilization plus a recommended `mem_mb`, `runtime`, and GPU
tier per stage.

## What each source is trusted for — and what it is NOT

Measured on Kempner, and the reason the tool blends three sources instead of
just parsing `jobstats`:

| Signal | Source | Why |
|---|---|---|
| GPU SM utilization, VRAM | **jobstats only** | `sacct`'s `gres/gpumem` and `gres/gpuutil` are **always 0** on this cluster. |
| Host RAM peak | **benchmark `max_rss`**, else jobstats | jobstats' host-mem is *sampled* and here **undersampled every stage by 1.5–4x** (MSA: 33 GB sampled vs 142 GB real). Sizing off jobstats alone would OOM-kill. |
| Runtime, time limit | jobstats / sacct | — |
| Short jobs (< ~2 min) | jobstats returns `nodes: {}` | No GPU data at all. The tool falls back to sacct `MaxRSS` and reports `gpu=None`, never a fake `gpu=0`. |

So: **jobstats is the right and only benchmark for GPU behaviour; it is not a
safe benchmark for host RAM.** The snakemake benchmark TSV (`max_rss`) is the
source of truth for memory, which is why `--output-dir` matters.

## Two caveats before you act on a recommendation

- **One run is not a benchmark.** The same stage (esmfold) measured 0.1% SM in a
  5-job trial and 100% SM in a 76-job run — the trial jobs did almost no real
  folding. A recommendation is only as representative as the run behind it.
  Prefer a full-scale run; treat tiny trials as plumbing checks.
- **Sampling misses brief peaks.** Both `MaxRSS` and SM% are sampled. Read every
  number as "no smaller than", never a tight upper bound. The tool keeps the
  `scaling_models.yaml` safety margins (mem ×1.3, time ×1.5) on top.

GPU choice keys off SM utilization, not VRAM fit: a compute-bound stage (boltz
at 94% SM) keeps its fast card even though its VRAM would fit on a smaller one —
shrinking it just makes it slower for longer. Only demonstrably idle stages
(< 10% SM) are recommended down.
