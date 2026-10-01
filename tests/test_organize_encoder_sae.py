"""SAE runs merge layers/types; ordinary encoder runs still replace a size."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def organize(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "workflow/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("organize_encoder_outputs").organize


def _put(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_sae_merges_types_and_layers_replacing_only_matching_files(organize, tmp_path):
    sequences = tmp_path / "sequences"
    scratch = tmp_path / "scratch"
    _put(sequences, "protein/sae/600M/all-layers/sae_0.pt", b"keep layer")
    _put(sequences, "protein/sae/600M/all-layers/sae_18.pt", b"replace layer")
    _put(sequences, "protein/sae/600M/mlp/sae_18.pt", b"keep type")
    _put(sequences, "protein/sae/300M/mlp/sae_12.pt", b"keep size")
    _put(scratch, "protein/600M/all-layers/sae_18.pt", b"new layer 18")
    _put(scratch, "protein/600M/all-layers/sae_36.pt", b"new layer 36")

    assert organize(str(scratch), str(sequences), "sae") == 1
    files = {str(p.relative_to(sequences)): p.read_bytes()
             for p in sequences.rglob("*.pt")}
    assert files == {
        "protein/sae/600M/all-layers/sae_0.pt": b"keep layer",
        "protein/sae/600M/all-layers/sae_18.pt": b"new layer 18",
        "protein/sae/600M/all-layers/sae_36.pt": b"new layer 36",
        "protein/sae/600M/mlp/sae_18.pt": b"keep type",
        "protein/sae/300M/mlp/sae_12.pt": b"keep size",
    }
    assert not (scratch / "protein/600M").exists()


def test_non_sae_encoder_keeps_replace_semantics(organize, tmp_path):
    sequences = tmp_path / "sequences"
    scratch = tmp_path / "scratch"
    _put(sequences, "protein/esmc/600M/obsolete.pt", b"old")
    _put(sequences, "protein/esmc/300M/outputs.pt", b"other size")
    _put(scratch, "protein/600M/outputs.pt", b"new")
    assert organize(str(scratch), str(sequences), "esmc") == 1
    assert not (sequences / "protein/esmc/600M/obsolete.pt").exists()
    assert (sequences / "protein/esmc/600M/outputs.pt").read_bytes() == b"new"
    assert (sequences / "protein/esmc/300M/outputs.pt").read_bytes() == b"other size"
