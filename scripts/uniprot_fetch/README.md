# uniprot_fetch — bulk UniProt → per-protein FASTAs

Standalone utility (own venv, own `requirements.txt`) that reads a results file (`.xlsx` or `.csv`), pulls each protein's sequence from the UniProt REST API, and writes **one `.fasta` per protein** into the output directory — ready to feed straight into `pipeline.input.fasta_dir`.

This subdir is intentionally isolated from the project's main `pyproject.toml` / `requirements-data.txt`. **Do not add it to the main env.**

## What you get

```
out/
├── P12345.fasta        # one file per UniProt accession
├── Q9Y6K9.fasta
├── A0A0B4J2F0.fasta
├── ...
├── .fetch_complete     # sentinel — touched only when ALL candidates succeed
├── .snakemake/         # snakemake state dir (only when run via the Snakefile)
└── _logs/
    ├── manifest.tsv    # row, accession, name, header, length, status, path
    ├── failed.txt      # accessions that failed this run (empty on full success)
    └── snakemake.log   # only when run via the Snakefile
```

Each `.fasta` is a single record:

```
>{IDX}|protein|
{SEQUENCE_ON_A_SINGLE_LINE}
```

`IDX` is a sequential 1-based integer (1, 2, … 9000). Always ≤5 chars for inputs up to 99,999 rows — Boltz refuses longer headers, which is why the protein name is **not** in the header. Filenames are still accession-based (`P09001.fasta`) so files are unique on disk and easy to navigate.

The full mapping `idx ↔ row ↔ accession ↔ short_name ↔ raw ProteinNames cell` is written to `out/_logs/manifest.tsv` — that's where you look up "what is sequence 4137?":

```
idx  row  accession  short_name  protein_names_raw  header  length  status  path
1    0    P09001     MRPL3       MRPL3_HUMAN        1       340     OK      .../P09001.fasta
2    1    P38646     GRP75       GRP75_HUMAN        2       679     OK      .../P38646.fasta
...
```

`idx` is deterministic from input row order (running counter of valid-accession rows), so resume keeps headers stable as long as the input file doesn't change.

## Resume / cache

The script skips accessions whose `<output>/<ACCESSION>.fasta` already exists. So:

- Crashing mid-run is fine — re-running only fetches the missing files.
- Writes are atomic (temp file + rename), so a crash mid-write can't leave a half-written `.fasta` that resume would treat as cached.
- The `.fetch_complete` sentinel is **only** touched when every candidate succeeded; failures wipe a stale sentinel so Snakemake will re-run.
- `_logs/failed.txt` lists accessions still failing — feed it back to a follow-up run if needed.

Pass `--no-resume` (CLI) or `--config no_resume=true` (Snakemake) to force refetch.

## Setup (do this once)

```bash
cd scripts/uniprot_fetch
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

The `.venv/` lives inside this folder and is gitignored at the repo root. (For local dev on a Mac with conda, `conda activate protforge && pip install -r requirements.txt` works too.)

## Recommended: run via Snakemake

The Snakefile under this folder wraps the script with the same conventions as the rest of the pipeline (`log:`, `retries: 3`, `--rerun-incomplete`).

```bash
cd scripts/uniprot_fetch

# Full 9k run, 8 concurrent HTTP workers
snakemake --cores 8 \
    --config input=/path/to/9k.xlsx output=./out header_row=1

# Dry run to see what would happen
snakemake -n --cores 1 --config input=/path/to/9k.xlsx output=./out

# Resume an interrupted run (idempotent — only refetches missing files)
snakemake --cores 8 --config input=... output=./out --rerun-incomplete

# Force everything to refetch
snakemake --cores 8 --config input=... output=./out no_resume=true --forceall
```

Snakemake config keys (all optional except `input`):

| key | default | meaning |
|---|---|---|
| `input` | (required) | `.xlsx` / `.csv` results file |
| `output` | `out` | directory for per-protein `.fasta` files |
| `header_row` | `0` | 0-indexed header row (use `1` for the GA xlsx) |
| `workers` | `8` | concurrent HTTP workers inside the script |
| `limit` | `-1` | max candidates; `-1` = all rows |
| `column` | (auto) | URL column name override |
| `name_column` | (auto) | protein-name column override |
| `no_resume` | `false` | `true` to refetch existing `.fasta` files |

## Or: run the script directly

```bash
python fetch_sequences.py \
  --input  /path/to/9k.xlsx \
  --output ./out \
  --header-row 1 \
  --workers 8 \
  -v
```

| flag | default | meaning |
|---|---|---|
| `--input` | (required) | `.xlsx` or `.csv` results file. |
| `--output` | (required) | Directory for per-protein FASTAs (created if missing). |
| `--column` | auto | UniProt URL/accession column override. Auto-detect scans columns and picks the first whose first 20 non-null values yield a UniProt accession. |
| `--name-column` | auto | Protein-name column override. Auto-detect prefers `*protein*name*`, else any `*name*`. |
| `--header-row` | `0` | 0-indexed header row. Pass `1` for the GA results xlsx (two header rows). |
| `--limit` | `-1` | Max candidate accessions; `-1` = all rows. Use a small value for smoke tests. |
| `--workers` | `8` | Concurrent HTTP workers (`ThreadPoolExecutor`). Use `1` for purely sequential. |
| `--timeout` | `15` | Per-request HTTP timeout (seconds). |
| `--no-resume` | off | Refetch even if `<accession>.fasta` already exists. |
| `-v` / `--verbose` | off | Per-row progress log via `tqdm.write` (doesn't break the bar). |

### Stage prints

```
[1/5] Reading <input.xlsx>
      loaded 9,000 rows
[2/5] Resolving columns
      URL/accession column : 'UniProt ID'
      protein-name column  : 'ProteinNames'
[3/5] Collecting up to all candidate(s) from 'UniProt ID'
      9000 candidate(s) collected
[4/5] Resume scan: 4521 cached on disk, 4479 to fetch
[5/5] Fetching 4479 sequence(s) from rest.uniprot.org with 8 worker(s)
UniProt: 100%|██████████| 4479/4479 [03:12<00:00, 23.3seq/s]
```

## Throughput

Threads, not processes — work is HTTP-bound and `requests.Session` is thread-safe. Observed ~23 seq/s at `--workers 8`, so **a 9k run from a cold cache is ~5–10 min**. After that, re-runs at `--workers 8` finish in seconds (just the disk-existence checks).

If you hit 429 (rate-limited), drop `--workers` to 4. UniProt REST is generous but 8 sustained workers × 9k requests over a few minutes is the upper end of polite.

## Bundling for cluster upload

```bash
tar czf fastas.tar.gz -C out .   # ~5 MB for 9k entries; ~9k tiny files compress well
scp fastas.tar.gz cluster:~/data/
```

One tarball uploads much faster than 9k tiny files over scp. Untar on the cluster into `pipeline.input.fasta_dir` and the MSA stage picks them up.

## Why hit the REST API instead of scraping the HTML page?

`https://rest.uniprot.org/uniprotkb/<ACC>.fasta` returns canonical FASTA in one call — no HTML parsing, no biopython, no API key. It also handles obsolete-accession redirects (HTTP 301) automatically via `requests`'s `allow_redirects=True`.

## A wrinkle with the GA xlsx — Excel HYPERLINK formulas

The `UniProt ID` column in `GA_20_results.xlsx` holds Excel formulas of the form `=HYPERLINK("https://www.uniprot.org/uniprot/P24752", "P24752")`. When pandas reads the file, it returns the **display text** (`"P24752"`), not the formula or the URL. So `extract_accession()` accepts both URLs and bare accessions. No special flag is needed.

## Behaviour & known limits

- A row whose value is neither a UniProt URL nor a bare accession is **skipped** with status `no-accession` (does *not* count toward `--limit`).
- A row whose accession returns 4xx (e.g. 404) is logged with status `http-...` and counts as a failure.
- Connection errors and 5xx responses are retried 3× with exponential backoff (1s → 2s → 4s) inside the script. Snakemake adds another `retries: 3` around the whole job.
- 429 (rate-limited) currently counts as 4xx and won't auto-retry — drop `--workers` if it bites.

## Tests

```bash
pytest -q
```

54 tests, all offline (HTTP mocked), run in well under a second. Coverage includes:

- Accession extraction from modern / legacy / query-string URLs, HYPERLINK formula text, and bare accessions.
- `short_name` truncation (`_` split, length cap, NaN handling).
- `header_name_for` integer-index headers (`>1|protein|` … `>9999|protein|`).
- Auto-detection of URL and name columns; xlsx with custom `--header-row`.
- FASTA response parsing; per-file atomic write; legacy combined `write_fasta`.
- HTTP retry on 5xx / connection errors; immediate fail on 4xx.
- End-to-end runs (URL fixture, bare-accession fixture, explicit columns).
- Concurrency: `--workers 8` writes every file and preserves manifest order.
- **Resume**: second run only refetches the missing files; `--no-resume` forces refetch.
- **Sentinel**: touched on full success; cleared on partial failure (so Snakemake re-runs).
- Failed-list and manifest contents.

## Files

```
scripts/uniprot_fetch/
├── README.md                 # this file
├── requirements.txt          # pandas, openpyxl, requests, tqdm, pytest
├── Snakefile                 # snakemake wrapper (retries, logging)
├── fetch_sequences.py        # the script
└── tests/
    ├── conftest.py
    └── test_fetch_sequences.py
```
