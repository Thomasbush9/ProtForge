"""Populate the host-mounted ProtForge model cache.

Run on a node with internet access before submitting containerized jobs:

    python scripts/download_models.py --cache-dir "$PROTFORGE_ROOT/models/hf"

The resulting cache should be bind-mounted read-only into the SIF at
`/models/hf` and used as `HF_HOME` by ESM-C and ESMFold rules.

Revision pinning
----------------
By default, both repos resolve to whatever `main` points at right now. The
resolved commit SHA is logged so you can copy it into a pinned config:

    python scripts/download_models.py \
        --esmfold-revision <sha> --esmc-revision <sha>

Pinning is the supply-chain control: it guarantees the next user (or the
next rebuild) gets bit-for-bit identical weights, instead of whatever
upstream pushed to HEAD in the meantime.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ESMFOLD_REPO = "facebook/esmfold_v1"
ESMC_REPO = "EvolutionaryScale/esmc-600m-2024-12"


def _default_cache_dir() -> Path | None:
    root = os.environ.get("PROTFORGE_ROOT")
    if not root:
        return None
    return Path(root).expanduser() / "models" / "hf"


def _load_token(token_file: Path | None) -> None:
    if token_file is None:
        return
    token = token_file.expanduser().read_text().strip()
    if token:
        os.environ["HF_TOKEN"] = token


def _resolve_revision(repo_id: str, revision: str) -> str:
    """Resolve a branch name (e.g. 'main') to its current commit SHA.

    Returning the SHA — rather than the branch — locks the snapshot in time:
    even if HEAD moves between this call and snapshot_download, we download
    what we resolved. The SHA is also printed so the user can pin it later.
    """
    if revision != "main" and len(revision) >= 7:
        # Already looks like a SHA (or a tag the user explicitly chose).
        # Don't second-guess them — pass through and let HF handle it.
        return revision

    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(repo_id=repo_id, revision=revision)
    resolved = info.sha
    print(f"Resolved {repo_id}@{revision} -> {resolved}", flush=True)
    return resolved


def _download_esmfold(revision: str) -> str:
    """Download ESMFold weights at the resolved revision. Returns the SHA."""
    from huggingface_hub import snapshot_download

    sha = _resolve_revision(ESMFOLD_REPO, revision)
    # ESMFold publishes pytorch_model.bin rather than safetensors. Pulling the
    # snapshot directly avoids constructing the model on the download node.
    snapshot_download(
        repo_id=ESMFOLD_REPO,
        revision=sha,
        allow_patterns=["pytorch_model.bin", "*.json", "*.txt", "*.model"],
    )
    return sha


def _download_esmc(revision: str) -> str:
    """Download ESM-C weights at the resolved revision. Returns the SHA."""
    from huggingface_hub import snapshot_download

    sha = _resolve_revision(ESMC_REPO, revision)
    # The ESM SDK's ESMC.from_pretrained("esmc_600m") resolves this snapshot.
    snapshot_download(repo_id=ESMC_REPO, revision=sha)
    return sha


def main() -> None:
    default_cache = _default_cache_dir()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache,
        help=(
            "HF_HOME root to populate. Defaults to "
            "$PROTFORGE_ROOT/models/hf when PROTFORGE_ROOT is set."
        ),
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=None,
        help="Optional file containing a read-only Hugging Face token.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("esmfold", "esmc", "all"),
        default=["all"],
        help="Models to download. Default: all.",
    )
    parser.add_argument(
        "--esmfold-revision",
        default="main",
        help=(
            f"Pin {ESMFOLD_REPO} to this revision (branch, tag, or commit SHA). "
            "Default 'main' resolves to the current HEAD SHA and logs it. "
            "Pass a SHA to lock the download."
        ),
    )
    parser.add_argument(
        "--esmc-revision",
        default="main",
        help=(
            f"Pin {ESMC_REPO} to this revision (branch, tag, or commit SHA). "
            "Default 'main' resolves to the current HEAD SHA and logs it. "
            "Pass a SHA to lock the download."
        ),
    )
    args = parser.parse_args()

    if args.cache_dir is None:
        raise SystemExit(
            "ERROR: provide --cache-dir or set PROTFORGE_ROOT "
            "(uses $PROTFORGE_ROOT/models/hf)."
        )

    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    # Belt-and-suspenders: this script needs network access. If the user
    # invoked us inside the SIF (where HF_HUB_OFFLINE defaults to 1), the
    # snapshot_download below would fail with "offline mode". Clear the flag
    # for this process.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    _load_token(args.token_file)

    selected = set(args.models)
    if "all" in selected:
        selected = {"esmfold", "esmc"}

    print(f"HF_HOME={cache_dir}", flush=True)
    resolved: dict[str, str] = {}
    if "esmfold" in selected:
        print(f"Downloading {ESMFOLD_REPO}@{args.esmfold_revision}...", flush=True)
        resolved[ESMFOLD_REPO] = _download_esmfold(args.esmfold_revision)
    if "esmc" in selected:
        print(f"Downloading {ESMC_REPO}@{args.esmc_revision}...", flush=True)
        resolved[ESMC_REPO] = _download_esmc(args.esmc_revision)

    hub_dir = cache_dir / "hub"
    total = sum(p.stat().st_size for p in hub_dir.rglob("*") if p.is_file()) if hub_dir.exists() else 0
    print(f"Model cache ready: {hub_dir} ({total / 1e9:.2f} GB)", flush=True)
    if resolved:
        print("Resolved revisions (pin these to lock the cache contents):", flush=True)
        for repo, sha in resolved.items():
            print(f"  {repo} @ {sha}", flush=True)


if __name__ == "__main__":
    main()
