#!/usr/bin/env python3
"""Write a per-run provenance manifest into the output dir.

Captures *how* a run was produced so a result can be reproduced or audited
later: git commit, the resolved config, the container images (path + size +
recorded sha256), and the resolved HuggingFace model commit SHAs.

Called from the Snakefile `onstart:` handler. Pure-Python and best-effort:
any single probe that fails is recorded as null rather than aborting the run.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str], cwd: str | None = None) -> str | None:
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=10
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_provenance(repo_dir: str) -> dict:
    """Commit SHA, branch, and dirty flag for the repo (all best-effort)."""
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir)
    status = _run(["git", "status", "--porcelain"], cwd=repo_dir)
    return {
        "commit": commit,
        "branch": branch,
        # status is "" when clean, non-empty when there are uncommitted changes
        "dirty": bool(status) if status is not None else None,
    }


def _sif_sha256(sif: Path) -> str | None:
    """Read a cached `<sif>.sha256` sidecar (written by containers/build.sh).

    We deliberately do NOT hash multi-GB SIFs at run start — the digest is
    computed once at build time. Missing sidecar -> null (size+mtime still
    pin the image well enough to detect a change).
    """
    sidecar = sif.with_suffix(sif.suffix + ".sha256")
    if not sidecar.is_file():
        return None
    first = sidecar.read_text().split()
    return first[0] if first else None


def container_provenance(containers: dict) -> dict:
    """Per-unique-SIF record: path, existence, size, mtime, recorded sha256."""
    out: dict[str, dict] = {}
    for key, raw in (containers or {}).items():
        if key == "runtime" or not raw:
            continue
        sif = Path(raw)
        if str(sif) in out:
            continue
        rec: dict = {"path": str(sif), "exists": sif.is_file()}
        if sif.is_file():
            st = sif.stat()
            rec["size_bytes"] = st.st_size
            rec["mtime"] = datetime.fromtimestamp(
                st.st_mtime, tz=timezone.utc
            ).isoformat()
            rec["sha256"] = _sif_sha256(sif)
        out[str(sif)] = rec
    return out


def model_provenance(cache_dirs: list[str]) -> dict:
    """Resolved HF model commit SHAs from `<cache>/hub/models--*/refs/main`."""
    models: dict[str, str] = {}
    for cache in cache_dirs:
        if not cache:
            continue
        hub = Path(cache) / "hub"
        if not hub.is_dir():
            continue
        for repo in sorted(hub.glob("models--*")):
            ref = repo / "refs" / "main"
            if ref.is_file():
                models[repo.name] = ref.read_text().strip()
    return models


def build_manifest(
    config: dict,
    *,
    repo_dir: str,
    runtime: str,
    started_at: str,
) -> dict:
    containers = config.get("containers", {})
    cache_dirs = [
        config.get("esmc", {}).get("cache_dir", ""),
        config.get("esmfold", {}).get("cache_dir", ""),
        config.get("openfold", {}).get("cache_dir", ""),
    ]
    pipeline = config.get("pipeline", {})
    return {
        "schema": "protforge/run_manifest@1",
        "started_at": started_at,
        "git": git_provenance(repo_dir),
        "stages_enabled": [k for k, v in pipeline.items() if v],
        "container_runtime": runtime,
        "containers": container_provenance(containers),
        "models": model_provenance(cache_dirs),
        "config": config,
    }


def write_manifest(
    output_dir: str,
    config: dict,
    *,
    repo_dir: str,
    runtime: str,
    started_at: str | None = None,
) -> Path:
    """Write run_manifest.json into output_dir; return its path. Never raises."""
    if started_at is None:
        started_at = datetime.now(tz=timezone.utc).isoformat()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run_manifest.json"
    manifest = build_manifest(
        config, repo_dir=repo_dir, runtime=runtime, started_at=started_at
    )
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return path
