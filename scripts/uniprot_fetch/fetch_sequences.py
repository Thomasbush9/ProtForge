"""Fetch protein sequences from UniProt for entries in an Excel/CSV file.

Reads a results file where one column contains a UniProt entry URL,
extracts the accession from each URL, fetches the canonical FASTA from
the UniProt REST API (rest.uniprot.org), and writes ONE FASTA FILE PER
PROTEIN into the output directory.

Output layout:

    <output>/{ACCESSION}.fasta              # one file per protein
    <output>/_logs/manifest.tsv             # row, accession, name, header, length, status, path
    <output>/_logs/failed.txt               # accessions that failed this run
    <output>/.fetch_complete                # sentinel — touched only if no failures

Each FASTA file contents (sequence on a single line):

    >{IDX}|protein|
    {SEQUENCE}

IDX is a sequential 1-based index assigned in input-row order. It
always fits in <=5 chars for inputs up to 99,999 rows (Boltz refuses
longer headers, which is why we don't put the protein name here).

The full mapping `idx <-> row <-> accession <-> short_name <-> raw
protein-name cell` is written to ``<output>/_logs/manifest.tsv`` —
that's where you look up "what is sequence 4137?".

Resume / cache:
    Files that already exist on disk are skipped. Re-running is cheap
    and only re-fetches missing or previously-failed accessions. IDX
    assignment is deterministic from the input file (running counter
    of valid-accession rows), so resume keeps existing headers stable
    as long as the input file doesn't change.

Usage:
    python fetch_sequences.py \\
        --input  /path/to/results.xlsx \\
        --output ./out \\
        --header-row 1 \\
        --workers 8 -v
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm


_ACC_PATTERN = (
    r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}"
)
UNIPROT_ACC_RE = re.compile(_ACC_PATTERN)
UNIPROT_BARE_ACC_RE = re.compile(r"^(?:" + _ACC_PATTERN + r")$")
UNIPROT_URL_RE = re.compile(
    r"https?://(?:www\.)?uniprot\.org/uniprot(?:kb)?/(" + _ACC_PATTERN + r")"
)
REST_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"

NAME_MAX_LEN = 5


def extract_accession(value) -> Optional[str]:
    """Return the UniProt accession from a cell value, or None if no match.

    Accepts:
      - Full UniProt URLs (``https://www.uniprot.org/uniprot/P12345``).
      - Excel HYPERLINK formula text containing such a URL.
      - Bare accessions (``P12345``, ``A0A0B4J2F0``). pandas read_excel
        evaluates ``=HYPERLINK("url","P12345")`` to just the display text
        ``"P12345"``, so this fallback is required for real result files.
    """
    if not isinstance(value, str):
        if value is None:
            return None
        value = str(value)
    m = UNIPROT_URL_RE.search(value)
    if m:
        return m.group(1)
    stripped = value.strip()
    if UNIPROT_BARE_ACC_RE.match(stripped):
        return stripped
    return None


def short_name(value, max_len: int = NAME_MAX_LEN) -> str:
    """Derive a short protein identifier (<= max_len chars) from a column value.

    Strips any species suffix after ``_`` (so ``THIL_HUMAN`` -> ``THIL``)
    and truncates the leading token to ``max_len`` chars. Empty / NaN
    values yield ``""``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    return s.split("_")[0][:max_len]


def header_name_for(idx: int) -> str:
    """Header name is a sequential 1-based integer index. Always <=5 chars
    for inputs up to 99,999 rows, which keeps Boltz happy.

    The mapping idx -> accession / short_name / row is recorded in
    ``<output>/_logs/manifest.tsv`` so you can always look up which
    protein an idx refers to.
    """
    return str(idx)


def autodetect_url_column(df: pd.DataFrame, sample: int = 20) -> str:
    """Return the first column whose first ``sample`` non-null values yield
    a recognisable UniProt accession (either embedded in a URL or as a bare
    accession). Raises ValueError on no match."""
    for col in df.columns:
        head = df[col].dropna().astype(str).head(sample)
        if any(extract_accession(v) for v in head):
            return col
    raise ValueError(
        "Could not auto-detect a UniProt accession column; pass --column explicitly."
    )


def autodetect_name_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristic: pick a column whose name contains 'protein' and 'name',
    else any column whose name contains 'name', else None.
    """
    cols = [str(c) for c in df.columns]
    for c in cols:
        cl = c.lower()
        if "protein" in cl and "name" in cl:
            return c
    for c in cols:
        if "name" in c.lower():
            return c
    return None


def parse_fasta_response(body: str) -> Tuple[str, str]:
    """Parse a single-record FASTA response. Returns (header, sequence)."""
    lines = body.strip().splitlines()
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"Not a FASTA response: {body[:80]!r}")
    header = lines[0][1:].strip()
    seq = "".join(line.strip() for line in lines[1:])
    return header, seq


def fetch_sequence(
    accession: str,
    timeout: float = 15.0,
    retries: int = 3,
    session: Optional[requests.Session] = None,
) -> Tuple[str, str]:
    """Fetch FASTA for ``accession`` from UniProt REST. Returns (header, seq).

    Retries connection errors and 5xx with exponential backoff (1s, 2s, 4s).
    4xx (e.g. 404) raises immediately — won't fix itself by retrying.
    """
    sess = session or requests.Session()
    url = REST_URL.format(acc=accession)
    delay = 1.0
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(url, timeout=timeout, allow_redirects=True)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        if 400 <= resp.status_code < 500:
            resp.raise_for_status()
        if resp.status_code >= 500:
            last_exc = requests.HTTPError(f"{resp.status_code} for {accession}")
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise last_exc
        resp.raise_for_status()
        return parse_fasta_response(resp.text)
    assert last_exc is not None
    raise last_exc


def read_input(path: Path, header_row: int = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, header=header_row)
    if suffix == ".csv":
        return pd.read_csv(path, header=header_row)
    raise ValueError(f"Unsupported file type: {suffix}")


def format_record(name: str, seq: str) -> str:
    """Render one FASTA record as a string: ``>{name}|protein|\\n{seq}\\n``."""
    return f">{name}|protein|\n{seq}\n"


def write_fasta(records: List[Tuple[str, str]], path: Path) -> int:
    """Write (name, seq) records as a single combined FASTA. Returns count.

    Kept for callers that want the legacy combined layout. ``main`` uses
    per-file output via :func:`write_one_fasta` instead.
    """
    with path.open("w") as f:
        for name, seq in records:
            f.write(format_record(name, seq))
    return len(records)


def fasta_path_for(out_dir: Path, accession: str) -> Path:
    """Per-protein FASTA path: ``<out_dir>/<ACCESSION>.fasta``."""
    return out_dir / f"{accession}.fasta"


def write_one_fasta(name: str, seq: str, path: Path) -> Path:
    """Write a single FASTA record to ``path``. Atomic via temp + rename so
    a crash mid-write can't leave a half-written file that resume would
    treat as cached."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        f.write(format_record(name, seq))
    tmp.replace(path)
    return path


def _resolve_column(args_value: Optional[str], df: pd.DataFrame, kind: str, autodetect):
    """Common --column / --name-column resolution. Returns column name or None."""
    if args_value:
        if args_value not in df.columns:
            raise SystemExit(
                f"ERROR: --{kind} column {args_value!r} not in {list(df.columns)}"
            )
        return args_value
    return autodetect(df)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, type=Path, help="Path to .xlsx or .csv results file")
    ap.add_argument("--output", required=True, type=Path, help="Directory to write per-protein FASTAs into")
    ap.add_argument(
        "--column",
        default=None,
        help="Name of column with UniProt URL (auto-detected if omitted)",
    )
    ap.add_argument(
        "--name-column",
        default=None,
        help="Name of column with the protein short name (auto-detected if omitted)",
    )
    ap.add_argument(
        "--header-row",
        type=int,
        default=0,
        help="0-indexed row to use as the header (default 0; pass 1 for the GA results xlsx)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Max candidate accessions to fetch (default: -1 = all rows). Use a small value for smoke tests.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent HTTP workers (default: 8). Use 1 for purely sequential.",
    )
    ap.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout, seconds (default: 15)")
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Refetch even if <accession>.fasta already exists (default: skip existing).",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    in_path = args.input.expanduser().resolve()
    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "_logs"
    logs_dir.mkdir(exist_ok=True)

    print(f"[1/5] Reading {in_path}", flush=True)
    df = read_input(in_path, header_row=args.header_row)
    print(f"      loaded {len(df):,} rows", flush=True)

    print("[2/5] Resolving columns", flush=True)
    try:
        url_col = _resolve_column(args.column, df, "column", autodetect_url_column)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"      URL/accession column : {url_col!r}", flush=True)

    if args.name_column and args.name_column not in df.columns:
        print(
            f"ERROR: --name-column {args.name_column!r} not in {list(df.columns)}",
            file=sys.stderr,
        )
        return 2
    name_col = args.name_column or autodetect_name_column(df)
    print(f"      protein-name column  : {name_col!r}", flush=True)

    limit_desc = "all" if args.limit < 0 else str(args.limit)
    print(f"[3/5] Collecting up to {limit_desc} candidate(s) from {url_col!r}", flush=True)
    # candidate fields: (row_idx, idx, accession, short, raw_protein_name, header)
    # idx is a 1-based running counter of valid-accession rows — deterministic
    # from the input file, so resume keeps headers stable.
    candidates: List[Tuple[int, int, str, str, str, str]] = []
    skipped_no_acc: List[Tuple[int, str]] = []
    next_idx = 1
    for row_idx, raw in df[url_col].items():
        if args.limit >= 0 and len(candidates) >= args.limit:
            break
        acc = extract_accession(raw if isinstance(raw, str) else str(raw))
        if acc is None:
            skipped_no_acc.append((row_idx, str(raw)))
            continue
        raw_name_val = df.loc[row_idx, name_col] if name_col is not None else acc
        raw_name_str = "" if raw_name_val is None or (
            isinstance(raw_name_val, float) and pd.isna(raw_name_val)
        ) else str(raw_name_val)
        short = short_name(raw_name_val)
        header = header_name_for(next_idx)
        candidates.append((row_idx, next_idx, acc, short, raw_name_str, header))
        next_idx += 1
    print(
        f"      {len(candidates)} candidate(s) collected"
        + (f", {len(skipped_no_acc)} row(s) skipped (no accession)" if skipped_no_acc else ""),
        flush=True,
    )

    # Partition: cached (already on disk) vs todo (need fetch)
    cached_idxs: List[int] = []
    todo_idxs: List[int] = []
    for i, (_row, _idx, acc, _short, _raw, _hdr) in enumerate(candidates):
        path = fasta_path_for(out_dir, acc)
        if (not args.no_resume) and path.exists() and path.stat().st_size > 0:
            cached_idxs.append(i)
        else:
            todo_idxs.append(i)
    print(
        f"[4/5] Resume scan: {len(cached_idxs)} cached on disk, "
        f"{len(todo_idxs)} to fetch",
        flush=True,
    )

    print(
        f"[5/5] Fetching {len(todo_idxs)} sequence(s) from rest.uniprot.org "
        f"with {args.workers} worker(s)",
        flush=True,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "ProtForge-uniprot-fetch/0.2"})

    # results[i] = (status, seq_len, path_or_empty)
    results: List[Tuple[str, int, str]] = [("", 0, "")] * len(candidates)
    for i in cached_idxs:
        _row, _idx, acc, _short, _raw, _hdr = candidates[i]
        path = fasta_path_for(out_dir, acc)
        try:
            seq_len = sum(
                len(ln.strip())
                for ln in path.read_text().splitlines()
                if ln and not ln.startswith(">")
            )
        except OSError:
            seq_len = 0
        results[i] = ("cached", seq_len, str(path))

    def _fetch_one(i: int):
        _row, _idx, acc, _short, _raw, hdr = candidates[i]
        try:
            _h, seq = fetch_sequence(acc, timeout=args.timeout, session=session)
            path = write_one_fasta(hdr, seq, fasta_path_for(out_dir, acc))
            return i, ("OK", len(seq), str(path))
        except requests.HTTPError as e:
            return i, (f"http-{e}", 0, "")
        except requests.RequestException as e:
            return i, (f"net-{type(e).__name__}", 0, "")
        except OSError as e:
            return i, (f"io-{type(e).__name__}-{e}", 0, "")

    workers = max(1, args.workers)
    if todo_idxs:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_fetch_one, i) for i in todo_idxs]
            for fut in tqdm(
                cf.as_completed(futures),
                total=len(futures),
                desc="UniProt",
                unit="seq",
            ):
                i, payload = fut.result()
                results[i] = payload
                if args.verbose:
                    row_idx, idx, acc, _short, _raw, hdr = candidates[i]
                    status, n, _path = payload
                    tqdm.write(
                        f"  row {row_idx} idx {idx}: {acc} -> >{hdr}|protein|  "
                        f"{'OK' if status == 'OK' else 'FAIL'}  "
                        f"{n} aa  {status if status != 'OK' else ''}".rstrip()
                    )

    # --- write manifest + failed list ---
    # Manifest doubles as the index file: idx -> row, accession, short_name,
    # full ProteinNames cell, header, file path. Look up "what is sequence
    # 4137?" here.
    manifest_path = logs_dir / "manifest.tsv"
    with manifest_path.open("w") as f:
        f.write("idx\trow\taccession\tshort_name\tprotein_names_raw\theader\tlength\tstatus\tpath\n")
        for i, (row_idx, idx, acc, short, raw_name, hdr) in enumerate(candidates):
            status, n, path = results[i]
            # TSV-safe: drop tabs/newlines from the raw protein-name cell.
            raw_clean = raw_name.replace("\t", " ").replace("\n", " ")
            f.write(f"{idx}\t{row_idx}\t{acc}\t{short}\t{raw_clean}\t{hdr}\t{n}\t{status}\t{path}\n")
        for row_idx, raw in skipped_no_acc:
            raw_clean = raw.replace("\t", " ").replace("\n", " ")
            f.write(f"\t{row_idx}\t\t\t{raw_clean}\t\t0\tno-accession\t\n")

    failed = [
        candidates[i][2]  # accession is now field 2 (after row_idx, idx)
        for i in range(len(candidates))
        if results[i][0] not in ("OK", "cached")
    ]
    failed_path = logs_dir / "failed.txt"
    failed_path.write_text("\n".join(failed) + ("\n" if failed else ""))

    n_ok     = sum(1 for s, _, _ in results if s == "OK")
    n_cached = sum(1 for s, _, _ in results if s == "cached")
    n_failed = len(failed)

    print()
    print(f"Wrote {n_ok} new record(s); {n_cached} already cached on disk.")
    print(f"Output dir : {out_dir}")
    print(f"Manifest   : {manifest_path}")
    if n_failed:
        print(
            f"  {n_failed} candidate(s) failed (see {failed_path}). "
            f"Re-run to retry — successful files are kept.",
            file=sys.stderr,
        )

    sentinel = out_dir / ".fetch_complete"
    if n_failed == 0 and (n_ok + n_cached) == len(candidates):
        sentinel.touch()
    else:
        # Don't leave a stale sentinel from a previous successful run if
        # this run had failures.
        if sentinel.exists():
            sentinel.unlink()

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
