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

## When the user wants embeddings or structures per mutant

The scan cannot give them: it never runs a mutant through the model, so there
are no per-mutant embeddings to save. Getting one means a real forward pass per
variant — `L x 19` of them, e.g. 4522 for a 238-residue protein, against 1 for
the scan.

That is a normal pipeline run, so generate the mutants as input and use the
regular stages:

```bash
bash bash_scripts/generate_satmut.sh --input wt.fasta --output-dir muts/ --dry-run
bash bash_scripts/generate_satmut.sh --input wt.fasta --output-dir muts/
```

Then point `input.fasta_dir` at `muts/` and enable `esmc` (embeddings) or
`esmfold`/`boltz` (structures). `--dry-run` first: always show the user the file
count before writing thousands of files. `--positions '1-50'` restricts the scan
to a region. `muts/index.csv` maps each file name to its mutation and sequence,
for joining predictions back to variants.

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
# single sequence  (--config is any config with container/cache/SLURM settings)
python -m webapp.satmut_cli submit --config config.yaml \
    --seq MSKGEELFTG... --name gfp --size 6B --out-dir /path/scans

# batch: one scan per FASTA in a directory (one sequence per file)
python -m webapp.satmut_cli submit --config config.yaml \
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

- **Resource sizing ignores `--size`.** The submit step always reads the
  `slurm.resources.esmc` block regardless of the model size requested
  (`webapp/satmut.py` reads `resources.esmc`, not `esmc_<size>`). A config tuned
  for 600M (e.g. ~13 GB / 20 min) will under-provision a **6B** scan — the ~12 GB
  weights can OOM during load or time out. For 6B, **copy the config** (e.g.
  `config.satmut.yaml`) and bump `slurm.resources.esmc` to ~80 GB / 30 min before
  submitting, so the live pipeline config stays untouched. Always confirm the
  `--mem`/`--time` in the `--dry-run` sbatch line before a real submit.
- `--size` defaults to `6B`; the host cache pointed at by `containers.esmc` /
  `esmc.cache_dir` must actually contain `hub/models--biohub--ESMC-6B` (a cache
  with only 600M will fail the GPU job).
- Sizes are per scan: deriving/viewing a `6B` scan and a `300M` scan are separate
  `--size` invocations against the same `--out-dir`.
- One sequence per FASTA file in `--fasta-dir`; file stem becomes the scan name.
- This is the standalone saturation-mutagenesis workflow, separate from the main
  pipeline config — it only borrows the container/cache/SLURM settings from a
  config you point at.
