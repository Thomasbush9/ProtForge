"""Fetch protein sequences from UniProt for entries in an Excel/CSV file.

Reads a results file where one column contains a UniProt entry URL,
extracts the accession from each URL, fetches the canonical FASTA from
the UniProt REST API (rest.uniprot.org), and writes a combined FASTA.

Output FASTA layout (one record per protein, sequence on a single line):

    >{NAME}|protein|
    {SEQUENCE}

where NAME is taken from the protein-name column, split on `_` and
truncated to 5 characters (so e.g. ``THIL_HUMAN`` -> ``THIL``,
``GRP75_HUMAN`` -> ``GRP75``, ``LRPAP1`` -> ``LRPAP``).

Usage:
    python fetch_sequences.py \\
        --input  /path/to/results.xlsx \\
        --output ./out \\
        --header-row 1 \\
        --limit  10 -v
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
    """Write (name, seq) records as FASTA, sequence on a single line.
    Returns count written."""
    with path.open("w") as f:
        for name, seq in records:
            f.write(format_record(name, seq))
    return len(records)


def _resolve_column(args_value: Optional[str], df: pd.DataFrame, kind: str, autodetect):
    """Common --column / --name-column resolution. Returns column name or None.

    ``kind`` is just for error messages. ``autodetect`` is the auto-detector
    for this column type; if it raises or returns None we propagate.
    """
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
    ap.add_argument("--output", required=True, type=Path, help="Directory to write FASTA into")
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
    ap.add_argument("--limit", type=int, default=10, help="Max candidate accessions to fetch (default: 10)")
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent HTTP workers (default: 8). Use 1 for purely sequential.",
    )
    ap.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout, seconds (default: 15)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    in_path = args.input.expanduser().resolve()
    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Reading {in_path}", flush=True)
    df = read_input(in_path, header_row=args.header_row)
    print(f"      loaded {len(df):,} rows", flush=True)

    print("[2/4] Resolving columns", flush=True)
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

    print(f"[3/4] Collecting up to {args.limit} candidate(s) from {url_col!r}", flush=True)
    candidates: List[Tuple[int, str, str]] = []  # (row_idx, accession, short_name)
    skipped_no_acc: List[Tuple[int, str]] = []   # (row_idx, raw_value)
    seen_names: set[str] = set()
    for idx, raw in df[url_col].items():
        if len(candidates) >= args.limit:
            break
        acc = extract_accession(raw if isinstance(raw, str) else str(raw))
        if acc is None:
            skipped_no_acc.append((idx, str(raw)))
            continue
        raw_name = df.loc[idx, name_col] if name_col is not None else acc
        name = short_name(raw_name) or acc[:NAME_MAX_LEN]
        if name in seen_names:
            name = (name[: NAME_MAX_LEN - 1] + str(len(seen_names)))[:NAME_MAX_LEN]
        seen_names.add(name)
        candidates.append((idx, acc, name))
    print(
        f"      {len(candidates)} candidate(s) collected"
        + (f", {len(skipped_no_acc)} row(s) skipped (no accession)" if skipped_no_acc else ""),
        flush=True,
    )

    print(
        f"[4/4] Fetching {len(candidates)} sequence(s) from rest.uniprot.org "
        f"with {args.workers} worker(s)",
        flush=True,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "ProtForge-uniprot-fetch/0.1 (test script)"})

    results: List[Optional[Tuple[str, int, str]]] = [None] * len(candidates)
    # results[i] = (status, seq_len, sequence_or_empty)

    def _fetch_one(i: int, acc: str, name: str):
        try:
            _hdr, seq = fetch_sequence(acc, timeout=args.timeout, session=session)
            return i, ("OK", len(seq), seq)
        except requests.HTTPError as e:
            return i, (f"http-{e}", 0, "")
        except requests.RequestException as e:
            return i, (f"net-{type(e).__name__}", 0, "")

    workers = max(1, args.workers)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_fetch_one, i, acc, name)
            for i, (_idx, acc, name) in enumerate(candidates)
        ]
        for fut in tqdm(
            cf.as_completed(futures),
            total=len(futures),
            desc="UniProt",
            unit="seq",
        ):
            i, payload = fut.result()
            results[i] = payload
            if args.verbose:
                idx, acc, name = candidates[i]
                status, n, _seq = payload
                tqdm.write(
                    f"  row {idx}: {acc} -> {name}  "
                    f"{'OK' if status == 'OK' else 'FAIL'}  "
                    f"{n} aa  {status if status != 'OK' else ''}".rstrip()
                )

    print()
    print(f"{'row':>5}  {'accession':<12}  {'name':<6}  {'len':>6}  status")
    print(f"{'-' * 5}  {'-' * 12}  {'-' * 6}  {'-' * 6}  {'-' * 30}")
    for i, (idx, acc, name) in enumerate(candidates):
        status, n, _seq = results[i]
        print(f"{idx:>5}  {acc:<12}  {name:<6}  {n:>6}  {status}")
    for idx, raw in skipped_no_acc:
        snippet = raw[:30] + ("..." if len(raw) > 30 else "")
        print(f"{idx:>5}  {'':<12}  {'':<6}  {0:>6}  no-accession ({snippet!r})")

    fasta_records = [
        (candidates[i][2], results[i][2])
        for i in range(len(candidates))
        if results[i][0] == "OK"
    ]
    fasta_path = out_dir / "sequences.fasta"
    n_written = write_fasta(fasta_records, fasta_path)
    n_failed = len(candidates) - n_written
    print()
    print(f"Wrote {n_written} record(s) to {fasta_path}")
    if n_failed:
        print(
            f"  ({n_failed} candidate(s) failed; see status column above. "
            f"Re-run with a larger --limit if you need more.)",
            file=sys.stderr,
        )

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
