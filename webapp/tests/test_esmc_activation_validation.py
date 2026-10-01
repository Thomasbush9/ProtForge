"""Offline activation preflight: layer bounds and one complete cached revision."""

import json

import pytest

from snake_helpers import esmc_activation_errors, parse_esmc_layer_selection


def _snapshot(tmp_path, *, size="6B", sae_type="all-layers", revision="current",
              available=(30, 60), present=(30, 60), main=True):
    suffix = "sae-k64-codebook16384" if sae_type == "all-layers" else "sae-mlp-k64-codebook131072"
    repo = tmp_path / "hub" / f"models--biohub--ESMC-{size}-{suffix}"
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"available_layers": list(available)}))
    for layer in present:
        (snapshot / f"layer_{layer}.safetensors").touch()
    if main:
        (repo / "refs").mkdir(exist_ok=True)
        (repo / "refs" / "main").write_text(revision)
    return snapshot


def _config(tmp_path, **sae):
    return {
        "pipeline": {"esmc": False},
        "esmc": {
            "models": ["6B"], "cache_dir": str(tmp_path),
            "sae": {"enabled": True, "layers": "all", **sae},
        },
    }


def test_layer_selection_strips_spaces_and_accepts_6b_depth():
    assert parse_esmc_layer_selection(" 0, 30, 60, 80 ", "6B") == [0, 30, 60, 80]
    assert parse_esmc_layer_selection(" ALL ", "300M") is None


@pytest.mark.parametrize("value", ["", "1,,2", "1,", "-1", "1.5", "1,01", "81"])
def test_invalid_layer_selection_is_rejected(value):
    with pytest.raises(ValueError):
        parse_esmc_layer_selection(value, "6B")


@pytest.mark.parametrize("size,depth", [("300M", 30), ("600M", 36), ("6B", 80)])
def test_hidden_layer_bounds_are_model_specific(size, depth):
    assert parse_esmc_layer_selection(str(depth), size) == [depth]
    with pytest.raises(ValueError):
        parse_esmc_layer_selection(str(depth + 1), size)


def test_selected_6b_layer60_does_not_require_other_shards(tmp_path):
    _snapshot(tmp_path, present=(60,))
    assert esmc_activation_errors(_config(tmp_path, layers=" 60 ")) == []


def test_all_requires_every_advertised_layer(tmp_path):
    _snapshot(tmp_path, present=(30,))
    errors = esmc_activation_errors(_config(tmp_path))
    assert any("layer_60.safetensors" in error for error in errors)


def test_layers_absent_from_model_metadata_are_rejected(tmp_path):
    _snapshot(tmp_path)
    errors = esmc_activation_errors(_config(tmp_path, layers="40"))
    assert any("does not advertise SAE layers [40]" in error for error in errors)


def test_current_snapshot_cannot_borrow_shards_from_old_revision(tmp_path):
    _snapshot(tmp_path, present=(30,))
    _snapshot(tmp_path, revision="old", main=False, present=(60,))
    errors = esmc_activation_errors(_config(tmp_path))
    assert any("current" in error and "layer_60.safetensors" in error for error in errors)


def test_ref_selects_complete_current_snapshot_despite_incomplete_old(tmp_path):
    _snapshot(tmp_path)
    _snapshot(tmp_path, revision="old", main=False, present=())
    assert esmc_activation_errors(_config(tmp_path)) == []


def test_absent_ref_allows_only_unambiguous_snapshot(tmp_path):
    _snapshot(tmp_path, main=False)
    assert esmc_activation_errors(_config(tmp_path)) == []
    _snapshot(tmp_path, revision="other", main=False)
    assert esmc_activation_errors(_config(tmp_path))


@pytest.mark.parametrize("filename", ["config.json", "layer_60.safetensors"])
def test_broken_symlink_does_not_satisfy_required_asset(tmp_path, filename):
    snapshot = _snapshot(tmp_path)
    (snapshot / filename).unlink()
    (snapshot / filename).symlink_to(snapshot / "missing-blob")
    assert any(filename in error for error in esmc_activation_errors(_config(tmp_path)))


def test_mlp_uses_its_own_repo(tmp_path):
    _snapshot(tmp_path)
    cfg = _config(tmp_path, sae_type="mlp")
    assert esmc_activation_errors(cfg)
    _snapshot(tmp_path, sae_type="mlp")
    assert esmc_activation_errors(cfg) == []


def test_empty_sae_subset_inherits_models(tmp_path):
    _snapshot(tmp_path)
    assert esmc_activation_errors(_config(tmp_path, sizes=[])) == []


def test_empty_model_selection_blocks_enabled_work(tmp_path):
    cfg = _config(tmp_path, sizes=[])
    cfg["esmc"]["models"] = []
    assert any("at least one" in error for error in esmc_activation_errors(cfg))
    cfg["pipeline"]["esmc"] = True
    cfg["esmc"]["sae"]["enabled"] = False
    assert any("esmc.models" in error for error in esmc_activation_errors(cfg))


def test_invalid_sizes_and_sae_types_are_reported(tmp_path):
    cfg = _config(tmp_path, sizes=["7B"], sae_type="unknown")
    errors = esmc_activation_errors(cfg)
    assert any("7B" in error for error in errors)
    assert any("sae_type" in error for error in errors)


def test_raw_layers_checked_for_every_active_model(tmp_path):
    cfg = {
        "pipeline": {"esmc": True},
        "esmc": {"models": ["300M", "6B"], "save_hidden_states": True,
                 "hidden_layers": "60", "hidden_pool": "mean", "hidden_dtype": "float16"},
    }
    errors = esmc_activation_errors(cfg)
    assert any("300M" in error and "0..30" in error for error in errors)
    cfg["esmc"]["models"] = ["6B"]
    assert esmc_activation_errors(cfg) == []


@pytest.mark.parametrize("key,value", [("hidden_pool", "max"), ("hidden_dtype", "int8")])
def test_invalid_raw_output_settings_block_only_active_exports(key, value):
    cfg = {"pipeline": {"esmc": True}, "esmc": {"save_hidden_states": True, key: value}}
    assert any(key in error for error in esmc_activation_errors(cfg))
    cfg["esmc"]["save_hidden_states"] = False
    assert esmc_activation_errors(cfg) == []


def test_disabled_outputs_ignore_stale_invalid_settings():
    cfg = {
        "pipeline": {"esmc": False},
        "esmc": {"models": [], "save_hidden_states": True, "hidden_layers": "-9",
                 "sae": {"enabled": False, "sae_type": "bad", "layers": "bad"}},
    }
    assert esmc_activation_errors(cfg) == []


def test_offcluster_planning_skips_cache_but_rejects_invalid_layers(tmp_path):
    cfg = _config(tmp_path, layers="60")
    assert esmc_activation_errors(cfg)
    assert esmc_activation_errors(cfg, check_cache=False) == []
    cfg["esmc"]["sae"]["layers"] = "81"
    assert any("0..80" in error for error in esmc_activation_errors(cfg, check_cache=False))
