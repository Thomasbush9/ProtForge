"""Tests for workflow/scripts/organize_msa_outputs.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflow" / "scripts"))
from organize_msa_outputs import (
    get_protein_id, is_empty_msa, organize_chunk, parse_fasta_header,
    rewrite_a3m_header, truncate_id,
)
from conftest import make_a3m, make_fasta


class TestParseFastaHeader:
    def test_simple(self, tmp_path):
        """Simple header: >PROTEIN with sequence MKTL."""
        fasta = make_fasta(tmp_path, "PROTEIN", "MKTL")
        prefix, seq = parse_fasta_header(fasta)
        assert prefix == "PROTEIN"
        assert seq == "MKTL"

    def test_with_pipes(self, tmp_path):
        """Pipes in header are replaced with underscores."""
        fasta = make_fasta(tmp_path, "test", "MK", header=">sp|P123|NAME")
        prefix, seq = parse_fasta_header(fasta)
        assert prefix == "sp_P123_NAME"
        assert seq == "MK"

    def test_invalid(self, tmp_path):
        """Raises ValueError for non-FASTA content."""
        bad = tmp_path / "bad.fasta"
        bad.write_text("NOT A FASTA\nMKTL\n")
        with pytest.raises(ValueError, match="Not a valid FASTA"):
            parse_fasta_header(bad)


class TestGetProteinId:
    def test_simple(self, tmp_path):
        """Extracts first part before pipe as ID."""
        fasta = make_fasta(tmp_path, "test", "MK", header=">myid|protein|path")
        assert get_protein_id(fasta) == "myid"

    def test_fallback(self, tmp_path):
        """Falls back to filename stem when header ID is empty."""
        fasta = tmp_path / "fallback.fasta"
        fasta.write_text(">|empty\nMK\n")
        assert get_protein_id(fasta) == "fallback"


class TestIsEmptyMsa:
    def test_empty_msa(self, tmp_path):
        """Query-only a3m (no homologs) is detected as empty."""
        a3m = make_a3m(tmp_path, "synth", homologs=False)
        assert is_empty_msa(a3m) is True

    def test_real_msa(self, tmp_path):
        """A3m with homologs is not empty."""
        a3m = make_a3m(tmp_path, "natural", homologs=True)
        assert is_empty_msa(a3m) is False


class TestTruncateId:
    def test_short_id_unchanged(self):
        seen = set()
        assert truncate_id("abc", seen) == "abc"
        assert "abc" in seen

    def test_five_char_id_unchanged(self):
        seen = set()
        assert truncate_id("sfGFP", seen) == "sfGFP"

    def test_long_id_truncated(self):
        seen = set()
        assert truncate_id("mCherry", seen) == "mCher"

    def test_collision_gets_suffix(self):
        seen = {"mCher"}
        result = truncate_id("mCherry", seen)
        assert len(result) == 5
        assert result != "mCher"
        assert result.startswith("mChe")


class TestRewriteA3mHeader:
    def test_rewrites_first_header(self, tmp_path):
        a3m = tmp_path / "test.a3m"
        a3m.write_text(">101\nMKTL\n>homolog\nMRTL\n")
        rewrite_a3m_header(a3m, "newid")
        lines = a3m.read_text().splitlines()
        assert lines[0] == ">newid"
        assert lines[2] == ">homolog"


class TestOrganizeChunk:
    def test_exact_match(self, tmp_path):
        """Matches a3m by protein_prefix (exact stem match)."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        fasta = make_fasta(input_dir, "protA", "MKTLAE")

        colab_dir = tmp_path / "colab"
        colab_dir.mkdir()
        make_a3m(colab_dir, "protA")

        file_list = tmp_path / "file_list.txt"
        file_list.write_text(f"{fasta.resolve()}\n")

        seq_dir = tmp_path / "sequences"
        processed, skipped = organize_chunk(str(file_list), str(colab_dir), str(seq_dir))

        assert processed == 1
        assert skipped == 0
        assert (seq_dir / "protA" / "msa" / "protA.a3m").exists()
        assert (seq_dir / "protA" / "protA.yaml").exists()

    def test_order_fallback(self, tmp_path):
        """Falls back to index-based matching when names don't match."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        fasta = make_fasta(input_dir, "myseq", "ACDE")

        colab_dir = tmp_path / "colab"
        colab_dir.mkdir()
        # a3m has a different name than the fasta
        make_a3m(colab_dir, "completely_different_name")

        file_list = tmp_path / "file_list.txt"
        file_list.write_text(f"{fasta.resolve()}\n")

        seq_dir = tmp_path / "sequences"
        processed, skipped = organize_chunk(str(file_list), str(colab_dir), str(seq_dir))

        assert processed == 1
        assert (seq_dir / "myseq" / "msa" / "completely_different_name.a3m").exists()

    def test_creates_yaml(self, tmp_path):
        """Verifies YAML structure: version, sequences, protein.id/sequence/msa."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        fasta = make_fasta(input_dir, "protX", "ACDEFG")

        colab_dir = tmp_path / "colab"
        colab_dir.mkdir()
        make_a3m(colab_dir, "protX")

        file_list = tmp_path / "file_list.txt"
        file_list.write_text(f"{fasta.resolve()}\n")

        seq_dir = tmp_path / "sequences"
        organize_chunk(str(file_list), str(colab_dir), str(seq_dir))

        yaml_content = (seq_dir / "protX" / "protX.yaml").read_text()
        assert "version: 1" in yaml_content
        assert "sequences:" in yaml_content
        assert "protein:" in yaml_content
        assert '"protX"' in yaml_content
        assert "sequence: ACDEFG" in yaml_content
        # Path must be absolute so YAMLs work when symlinked into boltz_chunks/
        # (Boltz resolves relative msa paths against the symlink's parent, not target)
        expected_a3m = (seq_dir / "protX" / "msa" / "protX.a3m").resolve()
        assert f"msa: {expected_a3m}" in yaml_content

    def test_copies_related_files(self, tmp_path):
        """Copies .sto and .hhr alongside .a3m."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        fasta = make_fasta(input_dir, "protA", "MKTL")

        colab_dir = tmp_path / "colab"
        colab_dir.mkdir()
        make_a3m(colab_dir, "protA")
        (colab_dir / "protA.sto").write_text("STO data")
        (colab_dir / "protA.hhr").write_text("HHR data")

        file_list = tmp_path / "file_list.txt"
        file_list.write_text(f"{fasta.resolve()}\n")

        seq_dir = tmp_path / "sequences"
        organize_chunk(str(file_list), str(colab_dir), str(seq_dir))

        msa_dir = seq_dir / "protA" / "msa"
        assert (msa_dir / "protA.a3m").exists()
        assert (msa_dir / "protA.sto").exists()
        assert (msa_dir / "protA.hhr").exists()

    def test_empty_msa_uses_msa_empty(self, tmp_path):
        """Synthetic proteins with no homologs get msa: empty in YAML."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        fasta = make_fasta(input_dir, "synth", "ACDEFG")

        colab_dir = tmp_path / "colab"
        colab_dir.mkdir()
        make_a3m(colab_dir, "synth", homologs=False)

        file_list = tmp_path / "file_list.txt"
        file_list.write_text(f"{fasta.resolve()}\n")

        seq_dir = tmp_path / "sequences"
        organize_chunk(str(file_list), str(colab_dir), str(seq_dir))

        yaml_content = (seq_dir / "synth" / "synth.yaml").read_text()
        assert "msa: empty" in yaml_content

    def test_truncates_long_id(self, tmp_path):
        """Protein IDs longer than 5 chars are truncated for Boltz compatibility."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        fasta = make_fasta(input_dir, "mCherry", "ACDEFG")

        colab_dir = tmp_path / "colab"
        colab_dir.mkdir()
        make_a3m(colab_dir, "mCherry")

        file_list = tmp_path / "file_list.txt"
        file_list.write_text(f"{fasta.resolve()}\n")

        seq_dir = tmp_path / "sequences"
        organize_chunk(str(file_list), str(colab_dir), str(seq_dir))

        yaml_content = (seq_dir / "mCherry" / "mCherry.yaml").read_text()
        assert '"mCher"' in yaml_content
        # a3m header must match the truncated id
        a3m_content = (seq_dir / "mCherry" / "msa" / "mCherry.a3m").read_text()
        assert a3m_content.startswith(">mCher\n")

    def test_rewrites_a3m_header(self, tmp_path):
        """a3m first header is rewritten to match the YAML protein id."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        fasta = make_fasta(input_dir, "protA", "MKTL")

        colab_dir = tmp_path / "colab"
        colab_dir.mkdir()
        # Simulate ColabFold internal index header
        a3m_path = colab_dir / "protA.a3m"
        a3m_path.write_text(">101\nMKTL\n>homolog1\nMRTL\n")

        file_list = tmp_path / "file_list.txt"
        file_list.write_text(f"{fasta.resolve()}\n")

        seq_dir = tmp_path / "sequences"
        organize_chunk(str(file_list), str(colab_dir), str(seq_dir))

        a3m_content = (seq_dir / "protA" / "msa" / "protA.a3m").read_text()
        assert a3m_content.startswith(">protA\n"), "First header should be rewritten to match YAML id"
        assert ">homolog1\n" in a3m_content, "Other headers should be unchanged"

    def test_skips_missing_fasta(self, tmp_path):
        """Warns and skips when FASTA file doesn't exist."""
        file_list = tmp_path / "file_list.txt"
        file_list.write_text("/nonexistent/path/missing.fasta\n")

        colab_dir = tmp_path / "colab"
        colab_dir.mkdir()

        seq_dir = tmp_path / "sequences"
        processed, skipped = organize_chunk(str(file_list), str(colab_dir), str(seq_dir))

        assert processed == 0
        assert skipped == 1
