"""CPU-only regressions for SAE input, cache selection, and saved outputs."""

import ast
import importlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sae():
    # Match the runner tests' AST loading convention: avoid importing GPU-only
    # transformers extensions while exercising the actual helper implementations.
    path = ROOT / "containers/esmc_extract_sae.py"
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign))]
    namespace = {"Path": Path, "json": json, "yaml": yaml}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_fasta_only_chunks_parse_without_msa(sae, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "workflow/scripts"))
    chunks = importlib.import_module("prepare_boltz_chunks")
    source = tmp_path / "fasta inputs"
    source.mkdir()
    (source / "alpha.fasta").write_text(">different header\nACD\nEF\n")
    (source / "beta.fa").write_text(">beta\nGHIK\n")
    files = chunks.find_yaml_files(str(source), include_fasta=True)
    directories = chunks.create_chunks(files, str(tmp_path / "chunk outputs"), 1)
    parsed = {}
    for directory in directories:
        parsed.update(sae["parse_inputs"](str(directory)))
    assert parsed == {"alpha": "ACDEF", "beta": "GHIK"}


def test_yaml_inputs_still_parse(sae, tmp_path):
    (tmp_path / "protein.yaml").write_text(
        "sequences:\n  - protein:\n      sequence: ACDEF\n")
    assert sae["parse_inputs"](str(tmp_path)) == {"protein": "ACDEF"}


@pytest.mark.parametrize("content", [">empty\n", ">one\nACD\n>two\nEFG\n"])
def test_fasta_rejects_empty_or_multiple_sequences(sae, tmp_path, content):
    (tmp_path / "protein.fa").write_text(content)
    with pytest.raises(ValueError):
        sae["parse_inputs"](str(tmp_path))


def test_input_stems_cannot_collide_across_formats(sae, tmp_path):
    (tmp_path / "protein.fa").write_text(">one\nACD\n")
    (tmp_path / "protein.yaml").write_text(
        "sequences:\n  - protein:\n      sequence: EFG\n")
    with pytest.raises(ValueError, match="Duplicate stem"):
        sae["parse_inputs"](str(tmp_path))


@pytest.fixture
def cache(tmp_path):
    repo = "biohub/ESMC-600M-sae-k64-codebook16384"
    repo_dir = tmp_path / ("models--" + repo.replace("/", "--"))
    current = repo_dir / "snapshots/current"
    current.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs/main").write_text("current\n")
    (current / "config.json").write_text(json.dumps({"available_layers": [0, 18, 36]}))
    for layer in (0, 18, 36):
        (current / f"layer_{layer}.safetensors").write_bytes(b"fixture")
    return str(tmp_path), repo, repo_dir, current


def test_spaced_layer_selection_is_validated_and_preserves_order(sae, cache):
    hub, repo, _, _ = cache
    assert sae["parse_layers"](" 36, 0, 18 ", "600M", hub, repo) == [36, 0, 18]


@pytest.mark.parametrize("spec", ["", "1,,2", "18,18", "-1", "37", "1; echo unsafe", "1"])
def test_invalid_or_untrained_layers_fail_before_loading(sae, cache, spec):
    hub, repo, _, _ = cache
    with pytest.raises(ValueError):
        sae["parse_layers"](spec, "600M", hub, repo)


def test_all_requires_every_advertised_shard_in_current_snapshot(sae, cache):
    hub, repo, repo_dir, current = cache
    assert sae["parse_layers"](" all ", "600M", hub, repo) == [0, 18, 36]
    (current / "layer_36.safetensors").unlink()
    old = repo_dir / "snapshots/old"
    old.mkdir()
    (old / "layer_36.safetensors").write_bytes(b"old shard is not current")
    with pytest.raises(FileNotFoundError, match="layer_36"):
        sae["parse_layers"]("all", "600M", hub, repo)


def test_stale_current_ref_does_not_fall_back(sae, cache):
    hub, repo, repo_dir, _ = cache
    (repo_dir / "refs/main").write_text("missing")
    with pytest.raises(FileNotFoundError):
        sae["parse_layers"]("all", "600M", hub, repo)


def test_ambiguous_snapshots_without_ref_fail(sae, cache):
    hub, repo, repo_dir, _ = cache
    (repo_dir / "refs/main").unlink()
    (repo_dir / "snapshots/other").mkdir()
    with pytest.raises(ValueError, match="snapshot"):
        sae["parse_layers"]("all", "600M", hub, repo)


def test_per_protein_saved_tensor_does_not_retain_batch_storage(sae, tmp_path):
    torch = pytest.importorskip("torch")
    sae["torch"] = torch
    batch = torch.arange(120, dtype=torch.float32).reshape(30, 4)
    selected = batch[2:5]
    sae["save_sae"](str(tmp_path), "protein", "600M", "mlp", {18: selected})
    tensor = torch.load(tmp_path / "protein/600M/mlp/sae_18.pt", weights_only=True)["activations"]
    assert torch.equal(tensor, selected)
    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
