"""Shared fixtures for uniprot_fetch tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_xlsx(tmp_path: Path) -> Path:
    """A 3-row xlsx with a UniProt URL column and a ProteinNames column."""
    df = pd.DataFrame(
        {
            "UniProt ID": [
                "https://www.uniprot.org/uniprot/P12345",
                "https://www.uniprot.org/uniprot/Q9Y6K9",
                "https://www.uniprot.org/uniprot/A0A0B4J2F0",
            ],
            "Genes": ["GA", "GB", "GC"],
            "ProteinNames": ["THIL_HUMAN", "GRP75_HUMAN", "LRPAP1"],
        }
    )
    path = tmp_path / "fixture.xlsx"
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def sample_xlsx_bare_accessions(tmp_path: Path) -> Path:
    """xlsx whose URL column holds bare accession strings, mirroring what
    pandas.read_excel returns for the real GA_20_results.xlsx (which uses
    Excel HYPERLINK formulas — pandas evaluates them to the display text).
    """
    df = pd.DataFrame(
        {
            "UniProt ID": ["P12345", "Q9Y6K9"],
            "ProteinNames": ["THIL_HUMAN", "GRP75_HUMAN"],
        }
    )
    path = tmp_path / "fixture_bare.xlsx"
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def fasta_body() -> str:
    return (
        ">sp|P12345|EXAMP_HUMAN Example protein OS=Homo sapiens OX=9606 GN=EXAMP PE=1 SV=2\n"
        "MENFKKLDSEAQRWRSLELRDQALAAATHLPHQHEQAARAEYHTLPLR\n"
        "QLAYDPLPLSESPALCPRSAFLAAEDQNAYLPHRGLLP\n"
    )
