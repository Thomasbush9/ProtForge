"""Tests for workflow/scripts/write_run_manifest.py"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflow" / "scripts"))
from write_run_manifest import (
    container_provenance,
    model_provenance,
    write_manifest,
)


def test_container_provenance_reads_sha_sidecar(tmp_path):
    sif = tmp_path / "gpu.sif"
    sif.write_bytes(b"fake image bytes")
    sif.with_suffix(".sif.sha256").write_text("deadbeef  gpu.sif\n")

    rec = container_provenance({"runtime": "singularity", "gpu": str(sif), "boltz": ""})

    # runtime + empty stage are skipped; the real SIF is recorded
    assert list(rec.keys()) == [str(sif)]
    entry = rec[str(sif)]
    assert entry["exists"] is True
    assert entry["size_bytes"] == len(b"fake image bytes")
    assert entry["sha256"] == "deadbeef"


def test_container_provenance_missing_sidecar_is_null(tmp_path):
    sif = tmp_path / "gpu.sif"
    sif.write_bytes(b"x")
    rec = container_provenance({"gpu": str(sif)})
    assert rec[str(sif)]["sha256"] is None


def test_container_provenance_dedupes_shared_sif(tmp_path):
    sif = tmp_path / "esm.sif"
    sif.write_bytes(b"x")
    # esmc and esmfold point at the same image — recorded once
    rec = container_provenance({"esmc": str(sif), "esmfold": str(sif)})
    assert len(rec) == 1


def test_model_provenance_reads_refs(tmp_path):
    repo = tmp_path / "hub" / "models--biohub--ESMC-600M" / "refs"
    repo.mkdir(parents=True)
    (repo / "main").write_text("abc123\n")

    models = model_provenance([str(tmp_path)])
    assert models == {"models--biohub--ESMC-600M": "abc123"}


def test_model_provenance_skips_missing_cache(tmp_path):
    assert model_provenance(["", str(tmp_path / "nope")]) == {}


def test_write_manifest_roundtrip(tmp_path):
    out = tmp_path / "outputs"
    config = {
        "pipeline": {"msa": True, "boltz": True, "esmc": False},
        "containers": {"runtime": "singularity"},
        "output": {"parent_dir": str(out)},
    }
    path = write_manifest(
        str(out), config, repo_dir=str(tmp_path), runtime="singularity",
        started_at="2026-06-30T00:00:00+00:00",
    )
    assert path == out / "run_manifest.json"

    data = json.loads(path.read_text())
    assert data["schema"] == "protforge/run_manifest@1"
    assert data["started_at"] == "2026-06-30T00:00:00+00:00"
    assert data["container_runtime"] == "singularity"
    # only enabled stages are listed
    assert set(data["stages_enabled"]) == {"msa", "boltz"}
    # the full resolved config is embedded for reproducibility
    assert data["config"] == config
