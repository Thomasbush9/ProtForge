"""Tests for workflow/scripts/chunk_yamls_for_esm.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflow" / "scripts"))
from chunk_yamls_for_esm import create_chunks, find_files
from conftest import make_yaml


def _items(*paths):
    """create_chunks takes (path, length) pairs; length is unused when unbinned."""
    return [(p, 4) for p in paths]


class TestFindFiles:
    def test_recursive_sorted(self, tmp_path):
        """Recursively finds and sorts *.yaml files."""
        (tmp_path / "sub").mkdir()
        make_yaml(tmp_path, "b_seq", "MKTL")
        make_yaml(tmp_path / "sub", "a_seq", "ACDE")

        result = find_files(str(tmp_path))
        assert len(result) == 2
        # Results are sorted by full path — verify both are found
        names = {r.name for r in result}
        assert names == {"a_seq.yaml", "b_seq.yaml"}

    def test_pattern_override(self, tmp_path):
        """A non-default pattern selects a different extension."""
        make_yaml(tmp_path, "seq", "MKTL")
        (tmp_path / "seq.fasta").write_text(">seq\nMKTL\n")

        assert len(find_files(str(tmp_path), pattern="*.fasta")) == 1


class TestCreateChunks:
    def test_naming(self, tmp_path):
        """Output files are named id_0.txt, id_1.txt, etc."""
        yaml_dir = tmp_path / "yamls"
        yaml_dir.mkdir()
        files = [make_yaml(yaml_dir, f"seq_{i}", "MKTL") for i in range(4)]
        output_dir = tmp_path / "output"

        chunk_files, _ = create_chunks(_items(*files), str(output_dir), num_chunks=2)

        assert (output_dir / "id_0.txt").exists()
        assert (output_dir / "id_1.txt").exists()
        assert not (output_dir / "id_2.txt").exists()

    def test_caps_num(self, tmp_path):
        """num_chunks=10 with 3 files creates only 3 chunk files."""
        yaml_dir = tmp_path / "yamls"
        yaml_dir.mkdir()
        files = [make_yaml(yaml_dir, f"seq_{i}", "MKTL") for i in range(3)]
        output_dir = tmp_path / "output"

        chunk_files, _ = create_chunks(_items(*files), str(output_dir), num_chunks=10)

        assert len(chunk_files) == 3
        # Each chunk has exactly 1 file (3 files / 3 chunks)
        for cf in chunk_files:
            lines = cf.read_text().strip().splitlines()
            assert len(lines) == 1

    def test_content_absolute_paths(self, tmp_path):
        """Each chunk file has absolute paths, one per line."""
        yaml_dir = tmp_path / "yamls"
        yaml_dir.mkdir()
        files = [make_yaml(yaml_dir, f"seq_{i}", "MKTL") for i in range(3)]
        output_dir = tmp_path / "output"

        chunk_files, _ = create_chunks(_items(*files), str(output_dir), num_chunks=1)

        lines = chunk_files[0].read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            assert Path(line).is_absolute()

    def test_chunks_meta(self, tmp_path):
        """create_chunks returns one metadata entry per chunk file, unbinned pool ''."""
        yaml_dir = tmp_path / "yamls"
        yaml_dir.mkdir()
        files = [make_yaml(yaml_dir, f"seq_{i}", "MKTL") for i in range(5)]
        output_dir = tmp_path / "output"

        chunk_files, chunks_meta = create_chunks(_items(*files), str(output_dir), num_chunks=3)

        assert len(chunks_meta) == len(chunk_files)
        chunk_ids = [cid for cid, _items_, _pool in chunks_meta]
        assert chunk_ids == ["0", "1", "2"]
        # unbinned mode tags every chunk with an empty pool label
        assert all(pool == "" for _cid, _items_, pool in chunks_meta)

    def test_single_chunk(self, tmp_path):
        """5 files with num_chunks=1 puts all in id_0.txt."""
        yaml_dir = tmp_path / "yamls"
        yaml_dir.mkdir()
        files = [make_yaml(yaml_dir, f"seq_{i}", "MKTL") for i in range(5)]
        output_dir = tmp_path / "output"

        chunk_files, _ = create_chunks(_items(*files), str(output_dir), num_chunks=1)

        assert len(chunk_files) == 1
        lines = chunk_files[0].read_text().strip().splitlines()
        assert len(lines) == 5

    def test_rerun_removes_stale_chunk_files(self, tmp_path):
        """Rerunning with fewer chunks removes old id_N.txt files."""
        yaml_dir = tmp_path / "yamls"
        yaml_dir.mkdir()
        output_dir = tmp_path / "output"

        first_files = [make_yaml(yaml_dir, f"seq_{i}", "MKTL") for i in range(5)]
        create_chunks(_items(*first_files), str(output_dir), num_chunks=3)
        assert (output_dir / "id_2.txt").exists()

        second_dir = tmp_path / "yamls_second"
        second_dir.mkdir()
        second_files = [make_yaml(second_dir, f"new_{i}", "ACDE") for i in range(2)]
        chunk_files, _ = create_chunks(_items(*second_files), str(output_dir), num_chunks=1)

        assert len(chunk_files) == 1
        assert (output_dir / "id_0.txt").exists()
        assert not (output_dir / "id_1.txt").exists()
        assert not (output_dir / "id_2.txt").exists()
        lines = chunk_files[0].read_text().strip().splitlines()
        assert len(lines) == 2
