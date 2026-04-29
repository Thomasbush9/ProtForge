# uniprot_fetch — exploratory script

Standalone utility (own `.venv`, own `requirements.txt`) that reads a results file (`.xlsx` or `.csv`), pulls each protein's sequence from the UniProt REST API, and writes a combined FASTA in a fixed layout.

This subdir is intentionally isolated from the project's main `pyproject.toml` / `requirements-data.txt`. **Do not add it to the main env.**

## Output FASTA layout

Matches `/Users/thom/tmp_data/GA_data/original.fasta`:

```
>{NAME}|protein|
{SEQUENCE_ON_A_SINGLE_LINE}
```

`NAME` is taken from the protein-name column (default: `ProteinNames`), split on `_` (so the species suffix is dropped) and truncated to 5 characters:

| ProteinNames value | NAME written |
|---|---|
| `THIL_HUMAN` | `THIL` |
| `GRP75_HUMAN` | `GRP75` |
| `MAOM_HUMAN` | `MAOM` |
| `LRPAP1` | `LRPAP` |
| `ACAT1` | `ACAT1` |

If two rows would produce the same `NAME`, the second occurrence is suffixed with a counter to keep records uniquely identifiable (e.g. `THIL` then `THI1`).

## Setup (do this once)

```bash
cd /Users/thom/Projects/ProtForge/scripts/uniprot_fetch
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

The `.venv/` lives inside this folder and is gitignored at the repo root.

## Run on the test file (10 entries)

```bash
python fetch_sequences.py \
  --input  /Users/thom/tmp_data/GA_data/GA_20_results.xlsx \
  --output ./out \
  --header-row 1 \
  --limit  10 -v
```

The GA results xlsx has **two header rows** — row 1 has merged category labels (`Protein Annotation`, `mean_log2_int`, …), row 2 has the actual column names (`UniProt ID`, `Genes`, `ProteinNames`, …). Pass `--header-row 1` to use the second row as the header.

Output:
- A per-row table on stdout: row index, accession, name, sequence length, status.
- `./out/sequences.fasta` containing the successful records in the layout above.

## CLI

| flag | default | meaning |
|---|---|---|
| `--input` | (required) | Path to `.xlsx` or `.csv` results file. |
| `--output` | (required) | Directory the FASTA goes into (created if missing). |
| `--column` | auto | Name of the column with the UniProt accession or URL. Auto-detect scans columns and picks the first one whose first 20 non-null values yield a UniProt accession (URL or bare). |
| `--name-column` | auto | Name of the column to derive the FASTA short name from. Auto-detect prefers a column matching `*protein*name*`, else any `*name*` column. |
| `--header-row` | 0 | 0-indexed row to use as the header. Pass `1` for the GA results xlsx. |
| `--limit` | 10 | Max **candidate accessions** to fetch. The script collects this many from the input and dispatches them all. |
| `--workers` | 8 | Concurrent HTTP workers (`ThreadPoolExecutor`). Use `1` for purely sequential. Output order always matches input row order regardless of worker count. |
| `--timeout` | 15 | Per-request HTTP timeout (seconds). |
| `-v` / `--verbose` | off | Per-row progress log (printed via `tqdm.write` so it doesn't break the bar). |

### Stage prints & progress bar

The script announces its four stages on stdout:

```
[1/4] Reading <input.xlsx>
      loaded 7,184 rows
[2/4] Resolving columns
      URL/accession column : 'UniProt ID'
      protein-name column  : 'ProteinNames'
[3/4] Collecting up to 10 candidate(s) from 'UniProt ID'
      10 candidate(s) collected
[4/4] Fetching 10 sequence(s) from rest.uniprot.org with 8 worker(s)
UniProt: 100%|██████████| 10/10 [00:00<00:00, 23.5seq/s]
```

### Performance

For a 10-entry run on `GA_20_results.xlsx`, observed locally:

| `--workers` | fetch wall time | notes |
|---|---|---|
| 1 | ~1.1 s | network-bound, sequential |
| 8 | ~0.4 s | ~2.5× faster on this batch; scales further at higher `--limit` |

Threads (not processes) are the right tool here — the work is HTTP-bound, not CPU-bound, and `requests.Session` is safe for concurrent use across threads.

## Why hit the REST API instead of scraping the HTML page?

`https://rest.uniprot.org/uniprotkb/<ACC>.fasta` returns canonical FASTA in one call — no HTML parsing, no biopython, no API key. It also handles obsolete-accession redirects (HTTP 301) automatically via `requests`'s `allow_redirects=True`.

## A wrinkle with the GA xlsx — Excel HYPERLINK formulas

The `UniProt ID` column in `GA_20_results.xlsx` holds Excel formulas of the form `=HYPERLINK("https://www.uniprot.org/uniprot/P24752", "P24752")`. When pandas reads the file, it returns the **display text** (`"P24752"`), not the formula or the URL. So `extract_accession()` accepts both URLs and bare accessions. No special flag is needed — just point `--column UniProt ID` (or let auto-detect find it).

## Behaviour & known limits

- A row whose value is neither a UniProt URL nor a bare accession is **skipped** with status `no-accession` (does *not* count toward `--limit`).
- A row whose accession returns 4xx (e.g. 404) is logged with status `http-...` and skipped.
- Connection errors and 5xx responses are retried 3× with exponential backoff (1s → 2s → 4s).
- No on-disk caching yet — re-runs re-fetch.
- Duplicate-name disambiguation truncates to keep the name within 5 chars; with many duplicates the suffix can collide. Fine for tens of rows, revisit if scaling.

## Tests

```bash
pytest -q
```

All tests are **offline** (HTTP mocked) and run in well under a second. They cover:

- Accession extraction from modern / legacy / query-string URLs, HYPERLINK formula text, and bare accessions.
- `short_name` for protein-name truncation (`_` split, length cap, NaN handling).
- Auto-detection of the URL column (URLs **and** bare-accession columns) and the name column.
- Reading xlsx with custom `--header-row`.
- FASTA response parsing.
- `format_record` / `write_fasta` produce the exact `>NAME|protein|\\n{seq}\\n` layout (single-line sequences).
- HTTP retry on 5xx / connection errors; immediate fail on 4xx.
- End-to-end pipeline runs (URL fixture, bare-accession fixture, explicit columns, duplicate-name disambiguation).
- Concurrency: a `--workers 8` run preserves input row order in the output FASTA.
- Partial-failure path: one 404 mid-batch yields rc=1 but the surviving records are still written.

## Verifying stage 2 manually

1. `pytest -q` passes (currently 48 tests).
2. Real run on `GA_20_results.xlsx` with `--limit 10 --header-row 1 -v` produces 10 `OK` rows.
3. `head -4 out/sequences.fasta` shows two records exactly: header line `>NAME|protein|`, then a single long sequence line. Compare side-by-side with `/Users/thom/tmp_data/GA_data/original.fasta`.
4. Spot-check one accession: copy the URL formula from Excel, fetch in a browser, confirm sequence matches.

## Files

```
scripts/uniprot_fetch/
├── README.md                 # this file
├── requirements.txt          # pandas, openpyxl, requests, pytest
├── fetch_sequences.py        # the script
└── tests/
    ├── conftest.py           # tiny xlsx fixtures + canned FASTA body
    └── test_fetch_sequences.py
```
