#!/usr/bin/env bash
# Populate HF cache for offline ESMFold2 container runs.
#
# Usage:
#   bash containers/download_scripts/esm_models.sh /path/to/cache_models

set -euo pipefail

CACHE_DIR="${1:-./esm_models_cache}"
ESMFOLD2_REPO="biohub/ESMFold2-Fast"
ESMC_REPO="biohub/ESMC-6B"

mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
echo "HF_HOME=$HF_HOME"

read -r -p "Provide a Hugging Face token? [y/N]: " REPLY
if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    read -r -p "Paste your Hugging Face token: " TOKEN
    export HF_TOKEN="$TOKEN"
fi

python - <<PY
from huggingface_hub import snapshot_download

repos = ("${ESMFOLD2_REPO}", "${ESMC_REPO}")
for repo in repos:
    print(f"Downloading {repo}...", flush=True)
    snapshot_download(repo_id=repo)
    print(f"[OK] {repo}", flush=True)

print(f"Cache ready: ${CACHE_DIR}/hub")
PY

echo "Download complete."
echo "Model cache ready: $CACHE_DIR"
