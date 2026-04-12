"""Tests for workflow/scripts/organize_msa_outputs.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflow" / "scripts"))
from organize_msa_outputs import get_protein_id, organize_chunk, parse_fasta_header
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
        assert "msa: msa/protX.a3m" in yaml_content
        # Path must be relative so YAMLs stay valid when sequences are copied/subset
        assert "/" + str(seq_dir) not in yaml_content, "MSA path should be relative, not absolute"

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
