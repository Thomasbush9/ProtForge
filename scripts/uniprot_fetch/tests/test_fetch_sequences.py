from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import fetch_sequences as fs  # noqa: E402


# --- accession extraction ---------------------------------------------------

@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.uniprot.org/uniprotkb/P12345/entry", "P12345"),
        ("https://www.uniprot.org/uniprotkb/P12345", "P12345"),
        ("https://www.uniprot.org/uniprot/P12345", "P12345"),
        ("https://www.uniprot.org/uniprotkb/A0A0B4J2F0/entry", "A0A0B4J2F0"),
        ("https://www.uniprot.org/uniprotkb/Q9Y6K9?xyz=1#section", "Q9Y6K9"),
        ("http://uniprot.org/uniprotkb/P12345", "P12345"),
        # Excel HYPERLINK formula text — regex still finds the URL inside.
        ('=HYPERLINK("https://www.uniprot.org/uniprot/P24752", "P24752")', "P24752"),
        # Bare accessions (what pandas read_excel returns for HYPERLINK cells).
        ("P12345", "P12345"),
        ("  Q9Y6K9  ", "Q9Y6K9"),
        ("A0A0B4J2F0", "A0A0B4J2F0"),
    ],
)
def test_extract_accession_valid(url, expected):
    assert fs.extract_accession(url) == expected


@pytest.mark.parametrize("bad", ["", "not a url", "https://example.com/foo", None, 42])
def test_extract_accession_invalid(bad):
    assert fs.extract_accession(bad) is None


# --- short_name -------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("THIL_HUMAN", "THIL"),
        ("GRP75_HUMAN", "GRP75"),
        ("MAOM_HUMAN", "MAOM"),
        ("LRPAP1", "LRPAP"),       # no underscore -> truncate to 5
        ("ACAT1", "ACAT1"),
        ("ABC", "ABC"),
        ("ABCDEFGH_HUMAN", "ABCDE"),  # long token before _ truncated
        ("", ""),
        (None, ""),
        (float("nan"), ""),
    ],
)
def test_short_name(raw, expected):
    assert fs.short_name(raw) == expected


def test_header_name_for():
    """Header is a sequential integer index — always <=5 chars at 9k scale,
    Boltz-friendly, with the idx -> accession mapping recorded in manifest.tsv."""
    assert fs.header_name_for(1) == "1"
    assert fs.header_name_for(42) == "42"
    assert fs.header_name_for(9999) == "9999"
    assert len(fs.header_name_for(99999)) == 5


# --- column auto-detection --------------------------------------------------

def test_autodetect_url_column_picks_url_col():
    df = pd.DataFrame({
        "name": ["a", "b"],
        "url": [
            "https://www.uniprot.org/uniprotkb/P12345/entry",
            "https://www.uniprot.org/uniprotkb/Q9Y6K9/entry",
        ],
    })
    assert fs.autodetect_url_column(df) == "url"


def test_autodetect_url_column_finds_url_in_hyperlink_formula():
    df = pd.DataFrame({
        "noise": ["a", "b"],
        "UniProt ID": [
            '=HYPERLINK("https://www.uniprot.org/uniprot/P24752", "P24752")',
            '=HYPERLINK("https://www.uniprot.org/uniprot/P23368", "P23368")',
        ],
    })
    assert fs.autodetect_url_column(df) == "UniProt ID"


def test_autodetect_url_column_missing_raises():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    with pytest.raises(ValueError):
        fs.autodetect_url_column(df)


def test_autodetect_name_column_prefers_protein_name():
    df = pd.DataFrame(columns=["UniProt ID", "Genes", "ProteinNames", "name_other"])
    assert fs.autodetect_name_column(df) == "ProteinNames"


def test_autodetect_name_column_falls_back_to_name():
    df = pd.DataFrame(columns=["a", "b", "FullName"])
    assert fs.autodetect_name_column(df) == "FullName"


def test_autodetect_name_column_returns_none():
    df = pd.DataFrame(columns=["a", "b"])
    assert fs.autodetect_name_column(df) is None


# --- input reading ----------------------------------------------------------

def test_read_input_xlsx(sample_xlsx):
    df = fs.read_input(sample_xlsx)
    assert len(df) == 3
    assert "UniProt ID" in df.columns


def test_read_input_unsupported(tmp_path):
    p = tmp_path / "weird.tsv"
    p.write_text("a\tb\n")
    with pytest.raises(ValueError):
        fs.read_input(p)


def test_read_input_header_row(tmp_path):
    """Mimics the GA results xlsx where the real header is on row 2."""
    df = pd.DataFrame(
        {
            "category_a": ["UniProt ID", "https://www.uniprot.org/uniprot/P12345"],
            "category_b": ["ProteinNames", "THIL_HUMAN"],
        }
    )
    p = tmp_path / "two_header.xlsx"
    df.to_excel(p, index=False)

    df_h0 = fs.read_input(p, header_row=0)
    assert "category_a" in df_h0.columns
    assert df_h0.iloc[0]["category_a"] == "UniProt ID"

    df_h1 = fs.read_input(p, header_row=1)
    assert "UniProt ID" in df_h1.columns
    assert df_h1.iloc[0]["UniProt ID"] == "https://www.uniprot.org/uniprot/P12345"


# --- FASTA parsing & writing ------------------------------------------------

def test_parse_fasta_response(fasta_body):
    header, seq = fs.parse_fasta_response(fasta_body)
    assert header.startswith("sp|P12345|EXAMP_HUMAN")
    assert "\n" not in seq
    assert seq.startswith("MENFKK")
    assert seq.endswith("RGLLP")


def test_parse_fasta_response_invalid():
    with pytest.raises(ValueError):
        fs.parse_fasta_response("not a fasta")


def test_format_record_layout():
    out = fs.format_record("1", "MENFKK")
    assert out == ">1|protein|\nMENFKK\n"


def test_write_one_fasta_atomic(tmp_path):
    """Verify per-file write produces the exact layout and no leftover .tmp."""
    long_seq = "M" * 250
    path = tmp_path / "P12345.fasta"
    written = fs.write_one_fasta("1", long_seq, path)
    assert written == path
    text = path.read_text()
    assert text == f">1|protein|\n{long_seq}\n"
    # No leftover atomic-write temp file
    assert not path.with_suffix(".fasta.tmp").exists()


def test_write_fasta_combined_still_works(tmp_path):
    """Legacy combined-output helper still produces single-line records."""
    long_seq = "M" * 250
    p = tmp_path / "out.fasta"
    fs.write_fasta([("1", long_seq), ("2", "ABCDEFG")], p)
    text = p.read_text()
    assert text == f">1|protein|\n{long_seq}\n>2|protein|\nABCDEFG\n"


# --- HTTP fetch with mocking ------------------------------------------------

def _mock_response(status: int, text: str = "") -> MagicMock:
    m = MagicMock()
    m.status_code = status
    m.text = text
    if status >= 400:
        m.raise_for_status.side_effect = requests.HTTPError(f"{status}")
    else:
        m.raise_for_status.return_value = None
    return m


def test_fetch_sequence_ok(fasta_body):
    sess = MagicMock()
    sess.get.return_value = _mock_response(200, fasta_body)
    header, seq = fs.fetch_sequence("P12345", session=sess)
    assert "P12345" in header
    assert seq.startswith("MENFKK")
    sess.get.assert_called_once()
    args, kwargs = sess.get.call_args
    assert "P12345.fasta" in args[0]
    assert kwargs["timeout"] == 15.0


def test_fetch_sequence_retries_on_5xx(fasta_body):
    sess = MagicMock()
    sess.get.side_effect = [
        _mock_response(503),
        _mock_response(503),
        _mock_response(200, fasta_body),
    ]
    with patch.object(fs.time, "sleep"):
        _, seq = fs.fetch_sequence("P12345", retries=3, session=sess)
    assert seq.startswith("MENFKK")
    assert sess.get.call_count == 3


def test_fetch_sequence_5xx_exhausts_retries():
    sess = MagicMock()
    sess.get.return_value = _mock_response(500)
    with patch.object(fs.time, "sleep"), pytest.raises(requests.HTTPError):
        fs.fetch_sequence("P12345", retries=2, session=sess)
    assert sess.get.call_count == 2


def test_fetch_sequence_404_does_not_retry():
    sess = MagicMock()
    sess.get.return_value = _mock_response(404)
    with patch.object(fs.time, "sleep"), pytest.raises(requests.HTTPError):
        fs.fetch_sequence("XXXXXX", retries=3, session=sess)
    assert sess.get.call_count == 1


def test_fetch_sequence_retries_on_connection_error(fasta_body):
    sess = MagicMock()
    sess.get.side_effect = [
        requests.ConnectionError("boom"),
        _mock_response(200, fasta_body),
    ]
    with patch.object(fs.time, "sleep"):
        _, seq = fs.fetch_sequence("P12345", retries=3, session=sess)
    assert seq.startswith("MENFKK")
    assert sess.get.call_count == 2


# --- end-to-end pipeline with mocked HTTP -----------------------------------

def _run_main_with_mocked_http(args_list, fasta_body, fake_get=None):
    """Run fs.main with a mocked requests.Session whose .get returns the given body."""
    if fake_get is None:
        def fake_get(url, *_, **__):
            return _mock_response(200, fasta_body)

    with patch.object(fs.requests, "Session") as session_cls, \
         patch.object(fs.time, "sleep"):
        sess = MagicMock()
        sess.get.side_effect = fake_get
        session_cls.return_value = sess
        return fs.main(args_list)


def _read_headers(out_dir: Path) -> dict[str, str]:
    """Return {accession: header_line} for every per-protein FASTA on disk."""
    headers: dict[str, str] = {}
    for p in sorted(out_dir.glob("*.fasta")):
        first = p.read_text().splitlines()[0]
        headers[p.stem] = first
    return headers


def test_pipeline_writes_per_protein_files(sample_xlsx, fasta_body, tmp_path, capsys):
    out_dir = tmp_path / "out"
    rc = _run_main_with_mocked_http(
        ["--input", str(sample_xlsx), "--output", str(out_dir),
         "--limit", "2", "--workers", "1"],
        fasta_body,
    )
    assert rc == 0
    headers = _read_headers(out_dir)
    # Filenames are accession-based (unique on disk); headers are 1-based idx.
    assert set(headers.keys()) == {"P12345", "Q9Y6K9"}
    assert headers["P12345"] == ">1|protein|"
    assert headers["Q9Y6K9"] == ">2|protein|"
    # Sentinel touched on full success
    assert (out_dir / ".fetch_complete").exists()
    # Manifest doubles as the idx -> protein lookup table.
    manifest = (out_dir / "_logs" / "manifest.tsv").read_text().splitlines()
    assert manifest[0] == "idx\trow\taccession\tshort_name\tprotein_names_raw\theader\tlength\tstatus\tpath"
    # idx=1 is P12345/THIL with the full ProteinNames cell preserved.
    fields = manifest[1].split("\t")
    assert fields[0] == "1"          # idx
    assert fields[2] == "P12345"     # accession
    assert fields[3] == "THIL"       # short_name
    assert fields[4] == "THIL_HUMAN" # raw protein-name cell
    assert fields[5] == "1"          # header
    assert fields[7] == "OK"
    out = capsys.readouterr().out
    assert "[1/5]" in out and "[5/5]" in out


def test_pipeline_explicit_columns(sample_xlsx, fasta_body, tmp_path):
    out_dir = tmp_path / "out"
    rc = _run_main_with_mocked_http(
        ["--input", str(sample_xlsx), "--output", str(out_dir),
         "--column", "UniProt ID", "--name-column", "ProteinNames",
         "--limit", "3", "--workers", "1"],
        fasta_body,
    )
    assert rc == 0
    headers = _read_headers(out_dir)
    assert headers == {
        "P12345":     ">1|protein|",
        "Q9Y6K9":     ">2|protein|",
        "A0A0B4J2F0": ">3|protein|",
    }


def test_pipeline_with_bare_accessions(sample_xlsx_bare_accessions, fasta_body, tmp_path):
    """Real GA_20_results.xlsx uses Excel HYPERLINK formulas; pandas evaluates
    them to bare accessions. The pipeline must handle that."""
    out_dir = tmp_path / "out"
    rc = _run_main_with_mocked_http(
        ["--input", str(sample_xlsx_bare_accessions), "--output", str(out_dir),
         "--limit", "2", "--workers", "1"],
        fasta_body,
    )
    assert rc == 0
    headers = _read_headers(out_dir)
    assert headers == {
        "P12345": ">1|protein|",
        "Q9Y6K9": ">2|protein|",
    }


def test_pipeline_concurrent_writes_all_files(sample_xlsx, fasta_body, tmp_path):
    """With workers > 1, every input row must still produce its own file."""
    out_dir = tmp_path / "out"
    rc = _run_main_with_mocked_http(
        ["--input", str(sample_xlsx), "--output", str(out_dir),
         "--limit", "3", "--workers", "8"],
        fasta_body,
    )
    assert rc == 0
    assert set(_read_headers(out_dir).keys()) == {"P12345", "Q9Y6K9", "A0A0B4J2F0"}
    # Manifest preserves input row order regardless of completion order.
    manifest_lines = (out_dir / "_logs" / "manifest.tsv").read_text().splitlines()[1:]
    idx_in_order = [line.split("\t")[0] for line in manifest_lines]
    accessions_in_order = [line.split("\t")[2] for line in manifest_lines]
    assert idx_in_order == ["1", "2", "3"]
    assert accessions_in_order == ["P12345", "Q9Y6K9", "A0A0B4J2F0"]


def test_pipeline_unknown_column_errors(sample_xlsx, tmp_path):
    rc = fs.main([
        "--input", str(sample_xlsx),
        "--output", str(tmp_path / "out"),
        "--column", "does_not_exist",
    ])
    assert rc == 2


def test_pipeline_resume_skips_existing(sample_xlsx, fasta_body, tmp_path):
    """A second run reuses files already on disk and only fetches the rest."""
    out_dir = tmp_path / "out"
    # First run: fetch 2 of 3.
    rc1 = _run_main_with_mocked_http(
        ["--input", str(sample_xlsx), "--output", str(out_dir),
         "--limit", "2", "--workers", "1"],
        fasta_body,
    )
    assert rc1 == 0
    assert (out_dir / "P12345.fasta").exists()
    assert (out_dir / "Q9Y6K9.fasta").exists()

    # Second run: limit 3, but the first two are cached -> only ONE HTTP call.
    call_count = {"n": 0}

    def counting_get(url, *_, **__):
        call_count["n"] += 1
        return _mock_response(200, fasta_body)

    rc2 = _run_main_with_mocked_http(
        ["--input", str(sample_xlsx), "--output", str(out_dir),
         "--limit", "3", "--workers", "1"],
        fasta_body,
        fake_get=counting_get,
    )
    assert rc2 == 0
    assert call_count["n"] == 1, "cached files must not be refetched"
    assert (out_dir / "A0A0B4J2F0.fasta").exists()
    # Manifest reflects mixed cached/OK statuses.
    manifest = (out_dir / "_logs" / "manifest.tsv").read_text()
    assert "cached" in manifest
    assert "OK" in manifest


def test_pipeline_no_resume_refetches(sample_xlsx, fasta_body, tmp_path):
    """--no-resume forces refetch of files already on disk."""
    out_dir = tmp_path / "out"
    rc1 = _run_main_with_mocked_http(
        ["--input", str(sample_xlsx), "--output", str(out_dir),
         "--limit", "1", "--workers", "1"],
        fasta_body,
    )
    assert rc1 == 0

    call_count = {"n": 0}

    def counting_get(url, *_, **__):
        call_count["n"] += 1
        return _mock_response(200, fasta_body)

    rc2 = _run_main_with_mocked_http(
        ["--input", str(sample_xlsx), "--output", str(out_dir),
         "--limit", "1", "--workers", "1", "--no-resume"],
        fasta_body,
        fake_get=counting_get,
    )
    assert rc2 == 0
    assert call_count["n"] == 1


def test_pipeline_partial_failure_no_sentinel(sample_xlsx, fasta_body, tmp_path):
    """One 404 in the batch -> rc=1, sentinel NOT written, failed.txt populated."""
    counter = {"n": 0}

    def fake_get(url, *_, **__):
        counter["n"] += 1
        if counter["n"] == 2:
            return _mock_response(404)
        return _mock_response(200, fasta_body)

    out_dir = tmp_path / "out"
    rc = _run_main_with_mocked_http(
        ["--input", str(sample_xlsx), "--output", str(out_dir),
         "--limit", "3", "--workers", "1"],
        fasta_body,
        fake_get=fake_get,
    )
    assert rc == 1
    # Two of three should have FASTA files on disk.
    assert len(list(out_dir.glob("*.fasta"))) == 2
    # No sentinel on partial failure.
    assert not (out_dir / ".fetch_complete").exists()
    # Failed accession recorded.
    failed = (out_dir / "_logs" / "failed.txt").read_text().strip().splitlines()
    assert len(failed) == 1


def test_pipeline_clears_stale_sentinel(sample_xlsx, fasta_body, tmp_path):
    """If a previous run touched the sentinel and a later run has failures,
    the stale sentinel must be removed so Snakemake re-runs the rule."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / ".fetch_complete").touch()

    def always_404(url, *_, **__):
        return _mock_response(404)

    rc = _run_main_with_mocked_http(
        ["--input", str(sample_xlsx), "--output", str(out_dir),
         "--limit", "1", "--workers", "1"],
        "",
        fake_get=always_404,
    )
    assert rc == 1
    assert not (out_dir / ".fetch_complete").exists()
