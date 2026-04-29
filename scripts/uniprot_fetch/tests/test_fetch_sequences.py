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
    out = fs.format_record("THIL", "MENFKK")
    assert out == ">THIL|protein|\nMENFKK\n"


def test_write_fasta_single_line_sequence(tmp_path):
    """Verify we DON'T wrap to 60 columns -- match original.fasta layout."""
    long_seq = "M" * 250
    p = tmp_path / "out.fasta"
    fs.write_fasta([("THIL", long_seq), ("GRP75", "ABCDEFG")], p)
    text = p.read_text()
    assert text == f">THIL|protein|\n{long_seq}\n>GRP75|protein|\nABCDEFG\n"
    # exactly 4 lines (2 headers + 2 single-line sequences)
    assert text.count("\n") == 4


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

def _run_main_with_mocked_http(args_list, fasta_body):
    """Run fs.main with a mocked requests.Session whose .get returns the given body."""
    def fake_get(url, *_, **__):
        return _mock_response(200, fasta_body)

    with patch.object(fs.requests, "Session") as session_cls:
        sess = MagicMock()
        sess.get.side_effect = fake_get
        session_cls.return_value = sess
        return fs.main(args_list)


def test_pipeline_with_limit(sample_xlsx, fasta_body, tmp_path, capsys):
    out_dir = tmp_path / "out"
    rc = _run_main_with_mocked_http(
        ["--input", str(sample_xlsx), "--output", str(out_dir),
         "--limit", "2", "--workers", "1"],
        fasta_body,
    )
    assert rc == 0
    fasta_text = (out_dir / "sequences.fasta").read_text()
    lines = fasta_text.splitlines()
    assert len(lines) == 4  # 2 records * (header + seq)
    assert lines[0] == ">THIL|protein|"
    assert lines[2] == ">GRP75|protein|"
    out = capsys.readouterr().out
    assert "[1/4]" in out and "[2/4]" in out and "[3/4]" in out and "[4/4]" in out


def test_pipeline_explicit_columns(sample_xlsx, fasta_body, tmp_path):
    out_dir = tmp_path / "out"
    rc = _run_main_with_mocked_http(
        ["--input", str(sample_xlsx), "--output", str(out_dir),
         "--column", "UniProt ID", "--name-column", "ProteinNames",
         "--limit", "3", "--workers", "1"],
        fasta_body,
    )
    assert rc == 0
    headers = [
        ln for ln in (out_dir / "sequences.fasta").read_text().splitlines()
        if ln.startswith(">")
    ]
    assert headers == [">THIL|protein|", ">GRP75|protein|", ">LRPAP|protein|"]


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
    headers = [
        ln for ln in (out_dir / "sequences.fasta").read_text().splitlines()
        if ln.startswith(">")
    ]
    assert headers == [">THIL|protein|", ">GRP75|protein|"]


def test_pipeline_concurrent_preserves_order(sample_xlsx, fasta_body, tmp_path):
    """With workers > 1, output order must still match input row order."""
    out_dir = tmp_path / "out"
    rc = _run_main_with_mocked_http(
        ["--input", str(sample_xlsx), "--output", str(out_dir),
         "--limit", "3", "--workers", "8"],
        fasta_body,
    )
    assert rc == 0
    headers = [
        ln for ln in (out_dir / "sequences.fasta").read_text().splitlines()
        if ln.startswith(">")
    ]
    assert headers == [">THIL|protein|", ">GRP75|protein|", ">LRPAP|protein|"]


def test_pipeline_unknown_column_errors(sample_xlsx, tmp_path):
    rc = fs.main([
        "--input", str(sample_xlsx),
        "--output", str(tmp_path / "out"),
        "--column", "does_not_exist",
    ])
    assert rc == 2


def test_pipeline_duplicate_names_disambiguated(tmp_path, fasta_body):
    """Two rows with the same protein short name -> second gets suffixed."""
    df = pd.DataFrame({
        "UniProt ID": [
            "https://www.uniprot.org/uniprot/P12345",
            "https://www.uniprot.org/uniprot/Q9Y6K9",
        ],
        "ProteinNames": ["THIL_HUMAN", "THIL_MOUSE"],
    })
    in_path = tmp_path / "dup.xlsx"
    df.to_excel(in_path, index=False)
    out_dir = tmp_path / "out"

    rc = _run_main_with_mocked_http(
        ["--input", str(in_path), "--output", str(out_dir),
         "--limit", "2", "--workers", "1"],
        fasta_body,
    )
    assert rc == 0
    headers = [
        ln for ln in (out_dir / "sequences.fasta").read_text().splitlines()
        if ln.startswith(">")
    ]
    assert headers[0] == ">THIL|protein|"
    assert headers[1] != headers[0]
    assert len(headers[1].split("|")[0].lstrip(">")) <= fs.NAME_MAX_LEN


def test_pipeline_partial_failure_returns_nonzero(sample_xlsx, fasta_body, tmp_path):
    """One 404 in the batch -> rc=1 but successful records still written."""
    counter = {"n": 0}

    def fake_get(url, *_, **__):
        counter["n"] += 1
        if counter["n"] == 2:
            return _mock_response(404)
        return _mock_response(200, fasta_body)

    out_dir = tmp_path / "out"
    with patch.object(fs.requests, "Session") as session_cls, \
         patch.object(fs.time, "sleep"):
        sess = MagicMock()
        sess.get.side_effect = fake_get
        session_cls.return_value = sess
        rc = fs.main([
            "--input", str(sample_xlsx), "--output", str(out_dir),
            "--limit", "3", "--workers", "1",
        ])
    assert rc == 1
    headers = [
        ln for ln in (out_dir / "sequences.fasta").read_text().splitlines()
        if ln.startswith(">")
    ]
    assert len(headers) == 2
