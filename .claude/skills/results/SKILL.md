---
name: results
description: >-
  Summarize and analyze a finished ProtForge run. Use when the user wants to
  analyze run results, show/compare structure confidences (pLDDT, pTM, ipTM,
  ranking), ask how long each stage took, see runtime / peak memory / node-hours
  per stage, or get an overall summary of a completed run. Reads the pipeline's
  output tree and Snakemake benchmark TSVs read-only via webapp/results_cli.py.
---

# Summarize a ProtForge run

Report on a completed (or in-progress) run: per-sequence structure confidences
and per-stage runtime/memory. The reading logic already exists in
`webapp/results.py` (the same functions the Streamlit Results tab uses) — this
skill drives it through the `webapp/results_cli.py` CLI.

Do **not** reimplement the readers. Call the CLI. It is strictly read-only.

## Picking the Python environment

`results_cli` needs `pyyaml` and `numpy` (for live confidence extraction). Run
from the repo root. If `python` is not on `PATH`, use the calibrate env, which
has both:

```bash
~/envs/protforge-calibrate/bin/python -m webapp.results_cli --help
```

(If the user activates their own env with these deps, use that instead.)

## Resolving the output dir

The CLI reads the output tree from, in order:
- `--output-dir /path` (explicit), or
- `--config config.<run>.yaml` → `output.parent_dir`.

Prefer `--config` when the user is working from a run config; fall back to
`--output-dir` if they hand you a path. The tree it reads:

```
<output.parent_dir>/sequences/{seq}/{boltz,openfold,esmfold}/...   # structures
<output.parent_dir>/benchmarks/{stage}/*.tsv                       # Snakemake benchmarks
```

## Commands

Both summaries (default):

```bash
~/envs/protforge-calibrate/bin/python -m webapp.results_cli --config config.<run>.yaml
```

Just one section:

```bash
# Structure confidences per sequence/model
~/envs/protforge-calibrate/bin/python -m webapp.results_cli --config config.<run>.yaml --structures
# Per-stage runtime / memory / node-hours
~/envs/protforge-calibrate/bin/python -m webapp.results_cli --config config.<run>.yaml --benchmarks
```

Machine-readable (parse it yourself instead of the table):

```bash
~/envs/protforge-calibrate/bin/python -m webapp.results_cli --config config.<run>.yaml --json
```

If the output dir doesn't exist yet, the CLI prints a clean "nothing to
summarize" message (exit 0) rather than erroring — the run probably hasn't
produced output. A large run can print hundreds of structure rows; for "how did
it do overall" prefer `--benchmarks`, or pipe `--json` and aggregate.

## Interpreting the output

**Structures table** — one row per (sequence, model), with headline confidences
normalized across predictors:
- **pLDDT** (0-100): per-residue local confidence, mean over the model.
  >90 very high, 70-90 confident, 50-70 low, <50 likely disordered.
- **pTM** (0-1): predicted TM-score, global fold confidence. >0.5 means a
  plausible overall fold; higher is better.
- **ipTM** (0-1): interface pTM for complexes. `0.000` on a single-chain
  monomer is expected (no interface), not a failure.
- **ranking**: the predictor's own composite ranking score for picking the best
  model.
- `-` means the predictor doesn't emit that metric (e.g. ESMFold reports pLDDT
  only; pTM/ipTM/ranking show `-`).

**Benchmarks table** — one row per pipeline stage (msa, boltz, esmc, esmc_sae,
esmfold, openfold), aggregated from the Snakemake benchmark TSVs:
- **jobs**: number of array jobs / rule invocations for that stage.
- **node·h**: node-hours = sum of wall-clock across that stage's jobs.
- **mean / p95 / max**: per-job wall-clock (minutes). A wide mean→max spread
  means uneven chunk sizes or stragglers.
- **peak mem**: max RSS across the stage's jobs (GB). Note MSA's peak is large
  (host RAM for the mmap'd ColabFold DB), which is expected — not an OOM.

The totals line gives overall job count and node-hours, useful for "what did
this run cost".

## Notes

- This skill is read-only — it never touches the output tree or config. For
  launching/sizing a run use the `run-pipeline` skill instead.
- Confidence values come from `<model>.summary.json` sidecars when present
  (written by the workflow's `organize_*` scripts) and fall back to live
  extraction from the CIF/JSON otherwise.
- Lab notes (run IDs, findings) live in the vault under
  `~/Documents/Vault/Notes/Lab/protforge/`, not the repo.
