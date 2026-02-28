"""Tests for workflow/scripts/prepare_boltz_chunks.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflow" / "scripts"))
from prepare_boltz_chunks import create_chunks, find_yaml_files
from conftest import make_yaml


class TestFindYamlFiles:
    def test_recursive(self, tmp_path):
        """Finds YAML files in nested directories."""
        (tmp_path / "sub1").mkdir()
        (tmp_path / "sub2" / "deep").mkdir(parents=True)
        make_yaml(tmp_path / "sub1", "a", "MKTL")
        make_yaml(tmp_path / "sub2" / "deep", "b", "ACDE")
        make_yaml(tmp_path, "c", "FGHI")

        result = find_yaml_files(str(tmp_path))
        assert len(result) == 3

    def test_empty(self, tmp_path):
        """Returns empty list for directory with no YAMLs."""
        assert find_yaml_files(str(tmp_path)) == []


class TestCreateChunks:
    def test_symlinks(self, tmp_path):
        """Created files are symlinks pointing to originals."""
        yaml_dir = tmp_path / "yamls"
        yaml_dir.mkdir()
        originals = [make_yaml(yaml_dir, f"seq_{i}", "MKTL") for i in range(3)]
        output_dir = tmp_path / "output"

        chunks = create_chunks(originals, str(output_dir), max_files_per_job=10)

        for f in chunks[0].iterdir():
            if f.name != "manifest.txt":
                assert f.is_symlink()
                assert f.resolve() in [o.resolve() for o in originals]

    def test_manifest(self, tmp_path):
        """manifest.txt has correct chunk directory paths."""
        yaml_dir = tmp_path / "yamls"
        yaml_dir.mkdir()
        files = [make_yaml(yaml_dir, f"seq_{i}", "MKTL") for i in range(5)]
        output_dir = tmp_path / "output"

        chunks = create_chunks(files, str(output_dir), max_files_per_job=2)

        manifest = (output_dir / "manifest.txt").read_text().strip().splitlines()
        assert len(manifest) == len(chunks)
        for line in manifest:
            assert Path(line).is_dir()

    def test_rerun_overwrites(self, tmp_path):
        """Running twice removes old symlinks and creates fresh ones."""
        yaml_dir = tmp_path / "yamls"
        yaml_dir.mkdir()
        files = [make_yaml(yaml_dir, f"seq_{i}", "MKTL") for i in range(2)]
        output_dir = tmp_path / "output"

        create_chunks(files, str(output_dir), max_files_per_job=10)
        # Run again — should not raise
        chunks = create_chunks(files, str(output_dir), max_files_per_job=10)

        symlinks = [f for f in chunks[0].iterdir() if f.is_symlink()]
        assert len(symlinks) == 2

    def test_chunk_sizes(self, tmp_path):
        """7 files with max_files=3 produces chunks of (3, 3, 1)."""
        yaml_dir = tmp_path / "yamls"
        yaml_dir.mkdir()
        files = [make_yaml(yaml_dir, f"seq_{i}", "MKTL") for i in range(7)]
        output_dir = tmp_path / "output"

        chunks = create_chunks(files, str(output_dir), max_files_per_job=3)

        assert len(chunks) == 3
        assert len(list(chunks[0].iterdir())) == 3
        assert len(list(chunks[1].iterdir())) == 3
        assert len(list(chunks[2].iterdir())) == 1
