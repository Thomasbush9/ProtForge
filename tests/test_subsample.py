"""Tests for scripts/calibrate/subsample.py — stratified FASTA subsampler."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "calibrate"))
from subsample import parse_fasta_length, stratified_sample
from conftest import make_fasta


class TestParseFastaLength:
    def test_single_sequence(self, tmp_path):
        p = make_fasta(tmp_path, "x", "ACDEFGHIK")
        assert parse_fasta_length(p) == 9

    def test_multi_line_sequence(self, tmp_path):
        p = tmp_path / "split.fasta"
        p.write_text(">seq\nACDE\nFGHI\nKL\n")
        assert parse_fasta_length(p) == 10

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.fasta"
        p.write_text("")
        assert parse_fasta_length(p) is None

    def test_header_only(self, tmp_path):
        p = tmp_path / "hdr.fasta"
        p.write_text(">just_header\n")
        assert parse_fasta_length(p) is None

    def test_strips_whitespace(self, tmp_path):
        p = tmp_path / "ws.fasta"
        p.write_text(">x\nAC DE\n  FG HI\n")
        assert parse_fasta_length(p) == 8


class TestStratifiedSample:
    def _items(self, lengths):
        # Use distinct fake paths so equality is by index, not content.
        return [(Path(f"f_{i}.fasta"), L) for i, L in enumerate(lengths)]

    def test_empty_input(self):
        assert stratified_sample([], n=10, seed=0) == []

    def test_zero_n(self):
        assert stratified_sample(self._items([100, 200]), n=0, seed=0) == []

    def test_picks_at_most_n(self):
        # Many items, request 5
        items = self._items(list(range(100, 200)))
        picked = stratified_sample(items, n=5, seed=42)
        assert len(picked) == 5

    def test_when_n_exceeds_input(self):
        # Asking for more than exists should just return everything available
        items = self._items([10, 20, 30])
        picked = stratified_sample(items, n=100, seed=0)
        assert len(picked) <= 100
        # All input items present
        assert {p[1] for p in picked} == {10, 20, 30}

    def test_spans_length_range(self):
        # Lengths uniformly distributed; sampler should span min to max
        items = self._items(list(range(50, 1551, 10)))  # 50, 60, ..., 1550
        picked = stratified_sample(items, n=20, seed=42)
        Ls = [L for _, L, _ in picked]
        # Should include something near the bottom and near the top
        assert min(Ls) <= 200, f"sample missed the short end: min={min(Ls)}"
        assert max(Ls) >= 1300, f"sample missed the long tail: max={max(Ls)}"

    def test_deterministic_with_seed(self):
        items = self._items(list(range(100, 1100)))
        a = stratified_sample(items, n=15, seed=7)
        b = stratified_sample(items, n=15, seed=7)
        assert [x[1] for x in a] == [x[1] for x in b]

    def test_different_seeds_differ(self):
        items = self._items(list(range(100, 1100)))
        a = stratified_sample(items, n=15, seed=1)
        b = stratified_sample(items, n=15, seed=2)
        # Not strictly required, but with 15 picks from 1000 items, two seeds
        # should almost always give different sets.
        assert [x[1] for x in a] != [x[1] for x in b]

    def test_output_sorted_by_length(self):
        items = self._items([300, 100, 1500, 600, 50, 1200])
        picked = stratified_sample(items, n=6, seed=0)
        lengths = [L for _, L, _ in picked]
        assert lengths == sorted(lengths)

    def test_bin_labels_present(self):
        items = self._items(list(range(100, 1100)))
        picked = stratified_sample(items, n=10, seed=0)
        labels = {label for _, _, label in picked}
        # At least 2 distinct buckets covered
        assert len(labels) >= 2
        for label in labels:
            assert label.startswith("q")


class TestSubsampleCLI:
    """Integration test: run the CLI end-to-end on a temp dir."""

    def test_writes_manifest_and_files(self, tmp_path):
        import subprocess

        src = tmp_path / "src"
        src.mkdir()
        for i, L in enumerate([76, 238, 585, 800, 1102, 1480]):
            make_fasta(src, f"seq_{i}", "A" * L)

        dest = tmp_path / "dest"
        repo_root = Path(__file__).resolve().parent.parent
        subsample_py = repo_root / "scripts" / "calibrate" / "subsample.py"
        result = subprocess.run(
            [sys.executable, str(subsample_py),
             "--input_dir", str(src),
             "--output_dir", str(dest),
             "--n", "4",
             "--seed", "0"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        manifest = dest / "manifest.csv"
        assert manifest.exists()

        rows = list(csv.DictReader(manifest.open()))
        assert len(rows) <= 4
        assert {"filename", "length", "bin", "source_path"} <= set(rows[0].keys())

        # Each manifest row corresponds to a real file in dest/
        for row in rows:
            f = dest / row["filename"]
            assert f.exists()
            assert int(row["length"]) > 0
