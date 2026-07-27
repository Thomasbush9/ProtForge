"""Tests for utils/generate_satmut.py — exhaustive single-point mutant FASTAs."""

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.generate_satmut import (  # noqa: E402
    parse_positions,
    satmut_variants,
    write_fastas,
    write_index,
)

WT = "MSKG"


class TestParsePositions:
    def test_none_means_every_position(self):
        assert parse_positions(None, 4) == [1, 2, 3, 4]

    def test_single_values_and_ranges_combine(self):
        assert parse_positions("1,3-5", 10) == [1, 3, 4, 5]

    def test_overlapping_ranges_deduplicate(self):
        assert parse_positions("1-3,2-4", 10) == [1, 2, 3, 4]

    def test_whitespace_and_empty_chunks_tolerated(self):
        assert parse_positions(" 1 , , 3 ", 5) == [1, 3]

    def test_position_past_the_end_is_rejected(self):
        """Silently scanning fewer positions than asked for would be worse."""
        with pytest.raises(ValueError, match="outside the sequence"):
            parse_positions("1-99", 4)

    def test_zero_is_rejected_since_positions_are_1_based(self):
        with pytest.raises(ValueError, match="outside the sequence"):
            parse_positions("0", 4)

    def test_backwards_range_is_rejected(self):
        with pytest.raises(ValueError, match="backwards"):
            parse_positions("5-2", 10)


class TestSatmutVariants:
    def test_skips_the_wild_type_residue(self):
        variants = satmut_variants(WT, [1], alphabet="MAC")
        muts = {m for _, m, _ in variants}
        assert muts == {"M1A", "M1C"}          # M1M omitted

    def test_count_is_positions_times_alphabet_minus_one(self):
        variants = satmut_variants(WT, [1, 2, 3, 4])
        assert len(variants) == 4 * 19

    def test_substitution_lands_at_the_right_index(self):
        _, _, seq = satmut_variants(WT, [2], alphabet="A")[0]
        assert seq == "MAKG"
        assert len(seq) == len(WT)

    def test_last_position_substitutes_correctly(self):
        _, mutation, seq = satmut_variants(WT, [4], alphabet="A")[0]
        assert mutation == "G4A"
        assert seq == "MSKA"

    def test_name_prefix_is_applied(self):
        name, _, _ = satmut_variants(WT, [1], alphabet="A", prefix="GFP_")[0]
        assert name == "GFP_M1A"

    def test_names_are_unique(self):
        variants = satmut_variants("MSKGMSKG", [1, 2, 3, 4, 5, 6, 7, 8])
        names = [n for n, _, _ in variants]
        assert len(names) == len(set(names))

    def test_original_sequence_is_not_mutated_in_place(self):
        satmut_variants(WT, [1, 2, 3, 4])
        assert WT == "MSKG"


class TestWriteOutputs:
    def test_writes_one_valid_single_record_fasta_each(self, tmp_path):
        variants = satmut_variants(WT, [1], alphabet="AC")
        write_fastas(variants, tmp_path)

        files = sorted(tmp_path.glob("*.fasta"))
        assert [f.name for f in files] == ["M1A.fasta", "M1C.fasta"]
        text = files[0].read_text()
        assert text.startswith(">M1A\n")
        assert text.endswith("\n")
        assert text.count(">") == 1

    def test_long_sequences_are_wrapped(self, tmp_path):
        long_wt = "A" * 150
        write_fastas(satmut_variants(long_wt, [1], alphabet="C"), tmp_path)
        body = (tmp_path / "A1C.fasta").read_text().splitlines()[1:]
        assert all(len(line) <= 60 for line in body)
        assert "".join(body) == "C" + "A" * 149

    def test_index_rows_match_the_fastas(self, tmp_path):
        variants = satmut_variants(WT, [1, 2], alphabet="AC")
        write_index(variants, tmp_path / "index.csv")

        with (tmp_path / "index.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(variants)
        first = rows[0]
        assert first["name"] == "M1A"
        assert first["mutation"] == "M1A"
        assert first["position"] == "1"
        assert first["wt_aa"] == "M"
        assert first["mut_aa"] == "A"
        assert first["sequence"] == "ASKG"

    def test_index_handles_multi_digit_positions(self, tmp_path):
        """mutation[1:-1] must give the whole number, not one digit."""
        long_wt = "A" * 150
        write_index(satmut_variants(long_wt, [123], alphabet="C"), tmp_path / "i.csv")
        with (tmp_path / "i.csv").open() as fh:
            row = next(csv.DictReader(fh))
        assert row["position"] == "123"
        assert row["mutation"] == "A123C"
