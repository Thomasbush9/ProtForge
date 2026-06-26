---
name: satmut
description: >-
  Zero-shot saturation-mutagenesis scanning of a wild-type protein with the
  ESM-C language model. Use when the user wants a saturation mutagenesis scan, a
  mutation scan / DMS heatmap, to know which mutations the model favors or
  disfavors, or a zero-shot variant-effect prediction. Scores every single-point
  substitution by log-likelihood ratio vs wild-type. Drives the submit -> derive
  -> view flow via webapp/satmut_cli.py. Batch-capable (one sequence or a dir of
  FASTAs).
---

# Saturation mutagenesis (zero-shot, ESM-C)

Score every single-point substitution of a wild-type sequence with ESM-C and
report which mutations the model favors vs disfavors. This is **analysis, not
folding**: there is one ESM-C forward pass per sequence, and the score is a
wild-type-marginal **log-likelihood ratio (LLR)**

    LLR(i, a) = logit(i, a) - logit(i, wt_i)

for substituting position `i` with amino acid `a`. **Negative LLR = the model
finds the mutation less likely than wild-type** (often destabilizing/deleterious);
positive = more likely than wild-type; the wild-type's own entry is exactly 0.
No mutant FASTAs are generated and nothing is folded.

The logic lives in `webapp/satmut.py` (launch/derive/load) and
`workflow/scripts/mutation_scan.py` (the LLR math). Do **not** reimplement it —
drive the `webapp/satmut_cli.py` CLI.

## Environment

The CLI needs only `numpy` + `pyyaml`. The deriving/viewing steps are pure NumPy
on the login node — no GPU, no torch. Only the GPU job (submitted by `submit`)
needs the ESM-C container. A suitable env:

```bash
~/envs/protforge-calibrate/bin/python   # has numpy + pyyaml
```

`python` may not be on PATH; use that interpreter (or the user's snakemake env).
Run from the repo root.

## Flow: submit -> (wait for GPU job) -> derive -> view

Three steps because the GPU forward pass is a SLURM job that takes time; deriving
and viewing happen after it lands.

### 1. submit — launch the GPU scan job(s)

Writes the query FASTA and submits a 1-GPU job that runs the ESM-C container with
`--save-logits`, producing `logits.npy` + `aa_token_ids.json`. Needs a config
with the container/cache/SLURM settings (same keys the pipeline uses:
`containers.esmc`, `esmc.cache_dir`, `slurm.partition/account/log_dir`).

```bash
# single sequence
python -m webapp.satmut_cli submit --config config.smoke.yaml \
    --seq MSKGEELFTG... --name gfp --size 6B --out-dir /path/scans

# batch: one scan per FASTA in a directory (one sequence per file)
python -m webapp.satmut_cli submit --config config.smoke.yaml \
    --fasta-dir /path/wts --size 600M --out-dir /path/scans

# preview the sbatch command without submitting (no cluster needed)
python -m webapp.satmut_cli submit ... --dry-run
```

`--size` is `6B` | `600M` | `300M` (default `6B`; 6B is most accurate, 300M
cheapest). Always `--dry-run` first to show the user the exact sbatch command.
If cluster settings are missing, the CLI lists them clearly; fix the config
before a real submit. After submit, the jobs run under SLURM — check with
`squeue -u $USER`. **Wait for the job(s) to finish before deriving.**

### 2. derive — compute the LLR matrix (after the job finishes)

Once `logits.npy` exists, derive the `Len x 20-AA` matrix into
`mutation_scan.csv`. Pure NumPy / CPU.

```bash
# every submitted scan for this size (skips & reports any not ready yet)
python -m webapp.satmut_cli derive --out-dir /path/scans --size 6B --all

# specific scans
python -m webapp.satmut_cli derive --out-dir /path/scans --size 6B --name gfp
```

It reads each wild-type sequence from the FASTA `submit` wrote, so you don't
re-pass it. Scans whose logits aren't ready are skipped and reported, not failed.

### 3. view — read the headline + CSV path

```bash
python -m webapp.satmut_cli view --out-dir /path/scans --name gfp --size 6B
python -m webapp.satmut_cli view --out-dir /path/scans --name gfp --size 6B --json
```

Prints the most/least mutation-tolerant positions (per-position mean LLR over the
19 non-WT substitutions — lower = less tolerant), the top favored single
substitutions, and the path to `mutation_scan.csv` (the full heatmap matrix for
plotting). `--json` for machine-readable output.

## Reading the output

- **LLR < 0**: model disfavors that substitution vs wild-type (likely
  deleterious). **LLR > 0**: model favors it.
- **Position sensitivity** = mean LLR over the 19 non-WT amino acids at a
  position. Low (very negative) = conserved / intolerant; high = tolerant.
- The full `Length x 20` matrix in `mutation_scan.csv` is the DMS-style heatmap:
  columns `position, wt_aa, sensitivity`, then one column per amino acid.

## Where outputs land

Under `--out-dir`:

```
input/<name>.fasta                       # query written by submit
<name>/<size>/logits.npy                 # GPU job output
<name>/<size>/aa_token_ids.json
<name>/<size>/mutation_scan.csv          # derive output (the heatmap)
<name>/<size>/scan_meta.json             # job id / submit metadata
```

## Notes

- Sizes are per scan: deriving/viewing a `6B` scan and a `300M` scan are separate
  `--size` invocations against the same `--out-dir`.
- One sequence per FASTA file in `--fasta-dir`; file stem becomes the scan name.
- This is the standalone saturation-mutagenesis workflow, separate from the main
  pipeline config — it only borrows the container/cache/SLURM settings from a
  config you point at.
