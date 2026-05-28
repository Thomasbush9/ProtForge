"""Populate the host-mounted ProtForge model cache.

Run on a node with internet access before submitting containerized jobs:

    python scripts/download_models.py --cache-dir "$PROTFORGE_ROOT/models/hf"

The resulting cache should be bind-mounted read-only into the SIF at
`/models/hf` and used as `HF_HOME` by ESM-C and ESMFold rules.
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


def _download_esmfold() -> None:
    from huggingface_hub import snapshot_download

    # ESMFold publishes pytorch_model.bin rather than safetensors. Pulling the
    # snapshot directly avoids constructing the model on the download node.
    snapshot_download(
        repo_id=ESMFOLD_REPO,
        allow_patterns=["pytorch_model.bin", "*.json", "*.txt", "*.model"],
    )


def _download_esmc() -> None:
    from huggingface_hub import snapshot_download

    # The ESM SDK's ESMC.from_pretrained("esmc_600m") resolves this snapshot.
    snapshot_download(repo_id=ESMC_REPO)


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
    args = parser.parse_args()

    if args.cache_dir is None:
        raise SystemExit(
            "ERROR: provide --cache-dir or set PROTFORGE_ROOT "
            "(uses $PROTFORGE_ROOT/models/hf)."
        )

    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    _load_token(args.token_file)

    selected = set(args.models)
    if "all" in selected:
        selected = {"esmfold", "esmc"}

    print(f"HF_HOME={cache_dir}", flush=True)
    if "esmfold" in selected:
        print(f"Downloading {ESMFOLD_REPO}...", flush=True)
        _download_esmfold()
    if "esmc" in selected:
        print(f"Downloading {ESMC_REPO}...", flush=True)
        _download_esmc()

    hub_dir = cache_dir / "hub"
    total = sum(p.stat().st_size for p in hub_dir.rglob("*") if p.is_file()) if hub_dir.exists() else 0
    print(f"Model cache ready: {hub_dir} ({total / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
