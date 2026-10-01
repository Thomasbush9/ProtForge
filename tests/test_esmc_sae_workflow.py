"""Exercise the real SAE checkpoint without running containers or model jobs."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


@pytest.mark.parametrize("source_kind", ["fasta", "yaml"])
def test_sae_checkpoint_reads_explicit_inputs_without_msa(tmp_path, source_kind):
    pytest.importorskip("snakemake")
    repo = Path(__file__).resolve().parents[1]
    (tmp_path / "workflow").symlink_to(repo / "workflow", target_is_directory=True)
    inputs = tmp_path / "input proteins"
    inputs.mkdir()
    if source_kind == "fasta":
        protein = inputs / "protein.fasta"
        protein.write_text(">protein\nACDE\n")
    else:
        protein = inputs / "protein.yaml"
        protein.write_text(yaml.safe_dump({
            "version": 1, "sequences": [{"protein": {"id": "A", "sequence": "ACDE"}}],
        }))
    cache = tmp_path / "cache"
    sae_repo = cache / "hub/models--biohub--ESMC-6B-sae-k64-codebook16384"
    snapshot = sae_repo / "snapshots/checkpoint-test"
    snapshot.mkdir(parents=True)
    (sae_repo / "refs").mkdir()
    (sae_repo / "refs/main").write_text(snapshot.name)
    (snapshot / "config.json").write_text(json.dumps({"available_layers": [30, 60]}))
    for layer in (30, 60):
        (snapshot / f"layer_{layer}.safetensors").touch()
    output = tmp_path / "output"
    sif = tmp_path / "esm.sif"
    sif.touch()  # Only the host checkpoint runs; no container is executed.
    config = {
        "pipeline": {"msa": False, "boltz": False, "esmc": False,
                     "esmfold": False, "openfold": False},
        "input": {f"{source_kind}_dir": str(inputs)},
        "output": {"parent_dir": str(output)},
        "esmc": {"models": ["6B"], "cache_dir": str(cache),
                 "sae": {"enabled": True, "layers": "30, 60"}},
        "slurm": {}, "containers": {"runtime": "singularity", "esmc": str(sif)},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    manifest = output / "esmc_sae_chunks/manifest.txt"
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "snakemake", "--snakefile", str(repo / "Snakefile"),
         "--directory", str(tmp_path), "--cores", "1", "--executor", "local", str(manifest)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    chunk = Path(manifest.read_text().strip())
    selected = list(chunk.iterdir())
    assert [p.resolve() for p in selected] == [protein.resolve()]
    assert not (output / "sequences").exists()
    assert not (output / ".esmc_sae_6B_complete").exists()
