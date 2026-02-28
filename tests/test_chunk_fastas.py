"""Tests for workflow/scripts/chunk_fastas.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflow" / "scripts"))
from chunk_fastas import create_chunks, find_fasta_files
from conftest import make_fasta


class TestFindFastaFiles:
    def test_both_extensions(self, tmp_path):
        """Finds both .fasta and .fa files, returned sorted."""
        make_fasta(tmp_path, "beta", "MKTL")
        (tmp_path / "alpha.fa").write_text(">alpha\nACDE\n")
        result = find_fasta_files(str(tmp_path))
        assert len(result) == 2
        assert result[0].name == "alpha.fa"
        assert result[1].name == "beta.fasta"

    def test_empty_dir(self, tmp_path):
        """Returns empty list for empty directory."""
        assert find_fasta_files(str(tmp_path)) == []

    def test_ignores_non_fasta(self, tmp_path):
        """Ignores .txt, .yaml, and other non-FASTA files."""
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "config.yaml").write_text("key: val")
        (tmp_path / "data.csv").write_text("a,b")
        make_fasta(tmp_path, "real", "MKTL")
        result = find_fasta_files(str(tmp_path))
        assert len(result) == 1
        assert result[0].name == "real.fasta"


class TestCreateChunks:
    def test_single_chunk(self, tmp_path):
        """3 files with max_files=5 produces 1 chunk."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        files = [make_fasta(input_dir, f"seq_{i}", "MKTL") for i in range(3)]
        output_dir = tmp_path / "output"

        chunks = create_chunks(files, str(output_dir), max_files_per_job=5)

        assert len(chunks) == 1
        assert (chunks[0] / "combined.fasta").exists()
        assert (chunks[0] / "file_list.txt").exists()

    def test_multiple_chunks(self, tmp_path):
        """5 files with max_files=2 produces 3 chunks (2,2,1)."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        files = [make_fasta(input_dir, f"seq_{i}", "MKTL") for i in range(5)]
        output_dir = tmp_path / "output"

        chunks = create_chunks(files, str(output_dir), max_files_per_job=2)

        assert len(chunks) == 3
        # Verify chunk sizes via file_list.txt line counts
        assert len((chunks[0] / "file_list.txt").read_text().strip().splitlines()) == 2
        assert len((chunks[1] / "file_list.txt").read_text().strip().splitlines()) == 2
        assert len((chunks[2] / "file_list.txt").read_text().strip().splitlines()) == 1

    def test_exact_division(self, tmp_path):
        """4 files with max_files=2 produces 2 chunks (2,2)."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        files = [make_fasta(input_dir, f"seq_{i}", "MKTL") for i in range(4)]
        output_dir = tmp_path / "output"

        chunks = create_chunks(files, str(output_dir), max_files_per_job=2)

        assert len(chunks) == 2
        assert len((chunks[0] / "file_list.txt").read_text().strip().splitlines()) == 2
        assert len((chunks[1] / "file_list.txt").read_text().strip().splitlines()) == 2

    def test_combined_fasta_content(self, tmp_path):
        """Combined FASTA contains all sequences with proper formatting."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        files = [
            make_fasta(input_dir, "protA", "AAAA"),
            make_fasta(input_dir, "protB", "BBBB"),
        ]
        output_dir = tmp_path / "output"

        chunks = create_chunks(files, str(output_dir), max_files_per_job=10)

        combined = (chunks[0] / "combined.fasta").read_text()
        assert ">protA" in combined
        assert "AAAA" in combined
        assert ">protB" in combined
        assert "BBBB" in combined

    def test_file_list_absolute_paths(self, tmp_path):
        """file_list.txt contains absolute paths."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        files = [make_fasta(input_dir, "seq_0", "MKTL")]
        output_dir = tmp_path / "output"

        chunks = create_chunks(files, str(output_dir), max_files_per_job=10)

        lines = (chunks[0] / "file_list.txt").read_text().strip().splitlines()
        for line in lines:
            assert Path(line).is_absolute()

    def test_manifest_lists_all_chunks(self, tmp_path):
        """manifest.txt has one line per chunk directory."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        files = [make_fasta(input_dir, f"seq_{i}", "MKTL") for i in range(5)]
        output_dir = tmp_path / "output"

        chunks = create_chunks(files, str(output_dir), max_files_per_job=2)

        manifest = (output_dir / "manifest.txt").read_text().strip().splitlines()
        assert len(manifest) == len(chunks)
        for line in manifest:
            assert Path(line).is_dir()
