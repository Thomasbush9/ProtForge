"""Tests for workflow/scripts/organize_boltz_outputs.py"""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflow" / "scripts"))
from organize_boltz_outputs import find_predictions_dir, organize_chunk
from conftest import make_boltz_prediction


class TestFindPredictionsDir:
    def test_exact(self, tmp_path):
        """Finds boltz_results_chunk_0/predictions/ by exact chunk ID."""
        pred = tmp_path / "boltz_results_chunk_0" / "predictions"
        pred.mkdir(parents=True)

        result = find_predictions_dir(str(tmp_path), "0")
        assert result == pred

    def test_fallback(self, tmp_path):
        """Falls back to any boltz_results_*/predictions/."""
        pred = tmp_path / "boltz_results_foobar" / "predictions"
        pred.mkdir(parents=True)

        result = find_predictions_dir(str(tmp_path), "99")
        assert result == pred

    def test_missing(self, tmp_path):
        """Returns None when no predictions directory exists."""
        result = find_predictions_dir(str(tmp_path), "0")
        assert result is None


class TestOrganizeChunk:
    def _setup_predictions(self, tmp_path, seq_id, model_numbers):
        """Helper: create boltz output structure with predictions dir."""
        boltz_dir = tmp_path / "boltz_output"
        results = boltz_dir / f"boltz_results_chunk_0" / "predictions"
        results.mkdir(parents=True)
        make_boltz_prediction(results, seq_id, model_numbers)
        return boltz_dir

    def test_picks_highest_model(self, tmp_path):
        """model_0, model_5, model_9 -> copies only model_9 files."""
        boltz_dir = self._setup_predictions(tmp_path, "protA", [0, 5, 9])
        seq_dir = tmp_path / "sequences"
        (seq_dir / "protA").mkdir(parents=True)

        organize_chunk(str(boltz_dir), "0", str(seq_dir))

        boltz_out = seq_dir / "protA" / "boltz"
        output_files = list(boltz_out.iterdir())
        assert len(output_files) == 2  # .cif + .json for model_9
        names = {f.name for f in output_files}
        assert "protA_model_9.cif" in names
        assert "protA_confidence_model_9.json" in names

    def test_handles_seq_prefix(self, tmp_path):
        """seq_123 dir in predictions -> target is seq_123/boltz/."""
        boltz_dir = self._setup_predictions(tmp_path, "seq_123", [0])
        seq_dir = tmp_path / "sequences"
        (seq_dir / "seq_123").mkdir(parents=True)

        organize_chunk(str(boltz_dir), "0", str(seq_dir))

        assert (seq_dir / "seq_123" / "boltz").is_dir()

    def test_handles_plain_id(self, tmp_path):
        """Plain ID '456' with no existing dir -> target is seq_456/boltz/."""
        boltz_dir = self._setup_predictions(tmp_path, "456", [0])
        seq_dir = tmp_path / "sequences"
        seq_dir.mkdir(parents=True)

        organize_chunk(str(boltz_dir), "0", str(seq_dir))

        assert (seq_dir / "seq_456" / "boltz").is_dir()

    def test_skips_no_model_files(self, tmp_path):
        """Warns and skips dirs with no _model_N files."""
        boltz_dir = tmp_path / "boltz_output"
        pred_dir = boltz_dir / "boltz_results_chunk_0" / "predictions" / "protA"
        pred_dir.mkdir(parents=True)
        # Create a file that doesn't match the model pattern
        (pred_dir / "some_other_file.txt").write_text("not a model")

        seq_dir = tmp_path / "sequences"

        processed = organize_chunk(str(boltz_dir), "0", str(seq_dir))
        assert processed == 0

    def test_delete_msa(self, tmp_path):
        """With delete_msa=True, removes msa/ directory."""
        boltz_dir = self._setup_predictions(tmp_path, "protA", [0])
        seq_dir = tmp_path / "sequences"
        msa_dir = seq_dir / "protA" / "msa"
        msa_dir.mkdir(parents=True)
        (msa_dir / "protA.a3m").write_text("dummy a3m")

        organize_chunk(str(boltz_dir), "0", str(seq_dir), delete_msa=True)

        assert not msa_dir.exists()
        assert (seq_dir / "protA" / "boltz").is_dir()

    def test_rerun_replaces_stale_model_files(self, tmp_path):
        """Rerunning removes old organized model files before copying new ones."""
        first_boltz_dir = self._setup_predictions(tmp_path / "first", "protA", [0, 9])
        seq_dir = tmp_path / "sequences"
        (seq_dir / "protA").mkdir(parents=True)

        organize_chunk(str(first_boltz_dir), "0", str(seq_dir))
        assert (seq_dir / "protA" / "boltz" / "protA_model_9.cif").exists()

        second_boltz_dir = self._setup_predictions(tmp_path / "second", "protA", [0])
        organize_chunk(str(second_boltz_dir), "0", str(seq_dir))

        boltz_out = seq_dir / "protA" / "boltz"
        names = {f.name for f in boltz_out.iterdir()}
        assert "protA_model_0.cif" in names
        assert "protA_confidence_model_0.json" in names
        assert "protA_model_9.cif" not in names
        assert "protA_confidence_model_9.json" not in names
