---
name: fetch-sequences
description: >-
  Ingest a spreadsheet of UniProt accessions/URLs into per-protein FASTA files
  for the ProtForge pipeline. Use when the user wants to fetch sequences from
  UniProt, ingest a spreadsheet of accessions, prepare FASTA inputs from a
  UniProt list, or turn an .xlsx/.csv of accessions into an input.fasta_dir.
  Interviews the researcher, runs the existing fetch CLI, verifies the output,
  and hands the directory off to the run-pipeline skill.
---

# Fetch sequences from UniProt

Drive a researcher from "I have a spreadsheet of UniProt entries" to a directory
of per-protein FASTAs ready to feed the pipeline's `input.fasta_dir`. The work
already exists in `scripts/uniprot_fetch/fetch_sequences.py` — this skill is the
conversational driver over it. Do **not** reimplement the fetch logic; call the
script.

The script reads an `.xlsx`/`.csv`, pulls each canonical FASTA from the UniProt
REST API, and writes **one `<ACCESSION>.fasta` per protein** plus a
`_logs/manifest.tsv` index and a `.fetch_complete` sentinel.

## Picking the Python environment

`python` is **not** on `PATH`. This utility is intentionally isolated from the
main project env and ships its own venv (see `scripts/uniprot_fetch/README.md`).
One-time setup:

```bash
cd scripts/uniprot_fetch
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If `.venv/` already exists, just activate it:

```bash
cd scripts/uniprot_fetch
source .venv/bin/activate
```

Run the fetch command below from inside `scripts/uniprot_fetch` with that venv
active. If activation fails or the deps are missing, do the one-time setup above
rather than guessing another interpreter.

## Procedure

### 1. Interview — pin down the inputs

Settle these before running. Ask only what you can't infer from the file itself
(column names are auto-detected, so confirm only when auto-detect is wrong):

- **Input spreadsheet** — path to the `.xlsx` or `.csv` results file (`--input`).
- **Accession/URL column** — which column holds the UniProt URL or bare
  accession. Auto-detected (first column whose first 20 values yield an
  accession); only set `--column` if auto-detect picks the wrong one.
- **Name column** — which column holds the protein short name. Auto-detected
  (prefers `*protein*name*`, else any `*name*`); only set `--name-column` to
  override. Used for the manifest, not the FASTA header.
- **Header row** — `--header-row` is 0-indexed (default `0`). Pass `1` for the
  GA results xlsx, which has two header rows.
- **Output dir** — where the per-protein FASTAs land (`--output`); created if
  missing. This becomes `input.fasta_dir`.
- **Workers** — `--workers` (default `8`) concurrent HTTP workers. Drop to `4`
  if UniProt returns 429 (rate-limited).
- **Smoke first?** — recommend `--limit N` (e.g. `--limit 20`) for a quick test
  before the full run; default `-1` fetches all rows.

### 2. Fetch

Verified command (run from `scripts/uniprot_fetch` with the venv active):

```bash
python fetch_sequences.py \
  --input  /path/to/results.xlsx \
  --output /path/to/out \
  --header-row 1 \
  --workers 8 \
  -v
```

Add `--column NAME` / `--name-column NAME` only to override auto-detect, and
`--limit N` for a smoke run. The script prints a `[1/5]…[5/5]` stage log and a
tqdm progress bar; a cold 9k run is ~5–10 min, re-runs finish in seconds.

### 3. Verify success

Check all three:

```bash
ls /path/to/out/.fetch_complete                 # sentinel — only touched on full success
head /path/to/out/_logs/manifest.tsv            # idx,row,accession,name,header,length,status,path
ls /path/to/out/*.fasta | wc -l                 # count of per-protein FASTAs
```

- The `.fetch_complete` sentinel exists **only** when every candidate
  succeeded; partial failures wipe it. The script exits non-zero on any failure.
- `_logs/failed.txt` lists accessions that failed this run (empty on full
  success). Re-run to retry — successful files are kept (see resume below).
- The FASTA count should match the candidate count from the `[3/5]` stage line
  (rows with no recognisable accession are skipped with status `no-accession`).

### 4. Hand off to the pipeline

The output dir is now a valid `input.fasta_dir`: one sequence per `.fasta`, the
format the MSA / sequence stages expect. Point the `run-pipeline` skill at it —
set `input.fasta_dir` in the run config to this directory. The `_logs/` and
`.fetch_complete` files are ignored by the pipeline scan.

To move FASTAs to the cluster, bundle them first (one tarball beats thousands of
tiny scp transfers), then untar into `input.fasta_dir`:

```bash
tar czf fastas.tar.gz -C /path/to/out .
```

## Notes

- **Resume / cache** — re-running skips accessions whose `<ACCESSION>.fasta`
  already exists, so a crash mid-run is cheap to recover: just re-run the same
  command and only missing/failed files are fetched. `idx` headers are
  deterministic from input-row order, so resume keeps them stable **as long as
  the input file doesn't change**. Pass `--no-resume` to force a full refetch.
- A Snakemake wrapper also exists (`scripts/uniprot_fetch/Snakefile`,
  `snakemake --cores 8 --config input=… output=… header_row=1`) with built-in
  `retries: 3` and logging — use it for large/unattended runs. Its
  `--config no_resume=true` mirrors the CLI `--no-resume`.
- FASTA headers are a bare 1-based integer index (`>1|protein|`), not the
  protein name — Boltz refuses long headers. The full
  `idx ↔ accession ↔ name` mapping lives in `_logs/manifest.tsv`; that's where
  you look up "what is sequence 4137?".
