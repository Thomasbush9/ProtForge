"""Model-free regressions for additive ESM-C raw states and layer exports."""

import ast
from argparse import ArgumentParser
import importlib.util
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
CONTAINERS = Path(__file__).resolve().parent.parent / "containers"


def _runner_namespace():
    """Extract runner logic without importing model-only transformers dependencies."""
    path = CONTAINERS / "run_batch_esmc.py"
    tree = ast.parse(path.read_text())
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign))]
    ns = {"torch": torch, "Path": Path, "time": time,
          "ArgumentParser": ArgumentParser}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), ns)
    main = next(node for node in tree.body if isinstance(node, ast.If))
    return ns, compile(ast.Module(body=main.body, type_ignores=[]), str(path), "exec")


@pytest.mark.parametrize("pool", ["none", "mean"])
def test_raw_states_preserve_standard_embeddings_from_same_forward(tmp_path, monkeypatch, pool):
    ns, main = _runner_namespace()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "short.fa").write_text(">short\nA\n")
    (inputs / "long.fa").write_text(">long\nAC\n")
    output = tmp_path / "outputs"
    stack = torch.arange(81 * 2 * 4 * 2, dtype=torch.float32).reshape(81, 2, 4, 2)
    calls = []

    class Model:
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(last_hidden_state=stack[-1], hidden_states=stack)

    def tokenizer(seqs, **kwargs):
        # The runner's real bucketing sorts these inputs by length.
        assert seqs == ["A", "AC"]
        return {
            "input_ids": torch.zeros((2, 4), dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
            "special_tokens_mask": torch.tensor([[1, 0, 1, 1], [1, 0, 0, 1]]),
        }

    ns["load_embed_model"] = lambda *args: (Model(), tokenizer)
    ns["_enforce_offline"] = lambda cache: None
    ns["_enable_tf32"] = lambda: None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", [
        "run_batch_esmc.py", "--cache", str(tmp_path / "cache"),
        "--input-dirs", str(inputs), "--output-dirs", str(output),
        "--size", "6B", "--save-hidden-states", "--hidden-layers", "60,30",
        "--hidden-pool", pool, "--hidden-dtype", "float16",
    ])
    exec(main, ns)

    assert len(calls) == 1, "raw states must reuse the embedding forward"
    for row, (name, length) in enumerate((("short", 1), ("long", 2))):
        directory = output / name / "6B"
        normal = torch.load(directory / "outputs.pt", weights_only=True)
        raw = torch.load(directory / "hidden_states.pt", weights_only=True)
        torch.testing.assert_close(normal["last_hidden_state"], stack[-1, row, 1:1 + length])
        expected = stack[[60, 30], row, 1:1 + length]
        expected = expected.mean(dim=1) if pool == "mean" else expected.half()
        torch.testing.assert_close(raw["hidden_states"], expected)
        assert raw["layers"] == [60, 30]
        assert raw["pool"] == pool
        assert raw["n_residues"] == length


@pytest.fixture
def exporter():
    spec = importlib.util.spec_from_file_location(
        "extract_esmc_layer", CONTAINERS / "extract_esmc_layer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_payload(pool="none"):
    stack = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    return {"hidden_states": stack.mean(dim=1) if pool == "mean" else stack,
            "layers": [60, 30], "pool": pool, "n_layers": 80, "n_residues": 3}


@pytest.mark.parametrize("pool", ["none", "mean"])
def test_export_uses_recorded_layer_ids_and_bounded_storage(tmp_path, exporter, pool):
    payload = _raw_payload(pool)
    src = tmp_path / "hidden_states.pt"
    torch.save(payload, src)
    directory, status = exporter.process_dir(str(src), None, False, 2)
    assert (directory, status) == (str(tmp_path), "ok")
    assert {p.name for p in tmp_path.glob("layer_*.pt")} == {"layer_60.pt", "layer_30.pt"}
    for row, layer_id in enumerate(payload["layers"]):
        layer = torch.load(tmp_path / f"layer_{layer_id}.pt", weights_only=True)
        torch.testing.assert_close(layer, payload["hidden_states"][row])
        assert layer.untyped_storage().nbytes() == layer.numel() * layer.element_size()


@pytest.mark.parametrize("metadata", [
    {"layers": None},
    {"layers": [30]},
    {"layers": [30, 30]},
    {"layers": [-1, 30]},
    {"layers": [True, 30]},
    {"layers": [30.0, 60]},
    {"layers": [30, 81]},
    {"pool": None},
    {"pool": "mean"},  # A pooled tensor cannot have a residue axis.
    {"n_layers": "80"},
    {"n_residues": 2},
])
def test_export_rejects_malformed_metadata_without_writing(tmp_path, exporter, metadata):
    payload = _raw_payload()
    payload.update(metadata)
    src = tmp_path / "hidden_states.pt"
    torch.save(payload, src)
    _, status = exporter.process_dir(str(src), None, False, None)
    assert status.startswith("error:")
    assert not list(tmp_path.glob("layer_*.pt"))


def test_export_rejects_missing_selection_metadata(tmp_path, exporter):
    src = tmp_path / "hidden_states.pt"
    torch.save({"hidden_states": _raw_payload()["hidden_states"]}, src)
    with pytest.raises(ValueError):
        exporter.load_stack(src)


def test_export_cli_defaults_to_raw_states_and_preserves_selected_ids(tmp_path):
    directory = tmp_path / "protein" / "6B"
    directory.mkdir(parents=True)
    embeddings = torch.zeros(3, 4)
    payload = _raw_payload("mean")
    torch.save({"last_hidden_state": embeddings}, directory / "outputs.pt")
    torch.save(payload, directory / "hidden_states.pt")
    subprocess.run(
        [sys.executable, str(CONTAINERS / "extract_esmc_layer.py"),
         str(tmp_path), "--workers", "1"],
        check=True, capture_output=True, text=True,
    )
    for row, layer_id in enumerate(payload["layers"]):
        torch.testing.assert_close(
            torch.load(directory / f"layer_{layer_id}.pt", weights_only=True),
            payload["hidden_states"][row],
        )
    torch.testing.assert_close(
        torch.load(directory / "outputs.pt", weights_only=True)["last_hidden_state"], embeddings,
    )
