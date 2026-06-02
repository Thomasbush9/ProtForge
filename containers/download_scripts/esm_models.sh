#!/usr/bin/env bash
# Populate HF cache for offline ESMFold2 container runs.
#
# Usage:
#   bash containers/download_scripts/esm_models.sh /path/to/cache_models

set -euo pipefail

CACHE_DIR="${1:-./esm_models_cache}"

model_repo() {
  case "$1" in
  ESMFold2) echo "biohub/ESMFold2" ;;
  ESMFold2_FAST) echo "biohub/ESMFold2-Fast" ;;
  ESMC-6B) echo "biohub/ESMC-6B" ;;
  ESMC-600M) echo "biohub/ESMC-600M" ;;
  ESMC-300M) echo "biohub/ESMC-300M" ;;
  *) return 1 ;;
  esac
}

model_sae_size() {
  case "$1" in
  ESMC-6B) echo "6B" ;;
  ESMC-600M) echo "600B" ;;
  ESMC-300M) echo "300M" ;;
  *) echo "" ;;
  esac
}
SAE_REPO_ALL_LAYERS="biohub/EMSC-XXXX-sae-k64-codebook16384"
SAE_REPO_MLP="biohub/EMSC-XXXX-sae-mlp-k64-codebook131072"

ask() {
  local reply
  read -rp "$1 [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
echo "HF_HOME=$HF_HOME"

read -r -p "Provide a Hugging Face token? [y/N]: " REPLY
if [[ "$REPLY" =~ ^[Yy]$ ]]; then
  read -r -p "Paste your Hugging Face token: " TOKEN
  export HF_TOKEN="$TOKEN"
fi

echo "Select the main models to instal..."
selected_models=()
for model in ESMFold2 ESMFold2_FAST ESMC-6B ESMC-600M ESMC-300M; do
  if ask "Install ${model} ($(model_repo "$model"))?"; then
    selected_models+=("$model")
  fi
done
selected_saes=()

for model in "${selected_models[@]}"; do
  size="$(model_sae_size "$model")"
  [[ -z "$size" ]] && continue

  repo="${SAE_REPO_ALL_LAYERS/XXXX/$size}"
  if ask "Install all-layers SAE for ${model} (${repo})?"; then
    selected_saes+=("$repo")
  fi

  repo="${SAE_REPO_MLP/XXXX/$size}"
  if ask "Install MLP SAE for ${model} (${repo})?"; then
    selected_saes+=("$repo")
  fi
done

echo "Installing:"
download_repos=()
echo "Models:"
for model in "${selected_models[@]}"; do
  repo="$(model_repo "$model")"
  printf '  - %s (%s)\n' "$model" "$repo"
  download_repos+=("$repo")
done
echo "SAEs:"
for repo in "${selected_saes[@]}"; do
  printf '  - %s\n' "$repo"
  download_repos+=("$repo")
done

export DOWNLOAD_REPOS_JSON
DOWNLOAD_REPOS_JSON="$(python -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${download_repos[@]}")"
python - <<'PY'
import json
import os

from huggingface_hub import snapshot_download

repos = json.loads(os.environ["DOWNLOAD_REPOS_JSON"])
cache_dir = os.environ["HF_HOME"]

if not repos:
    print("Nothing selected; cache unchanged.", flush=True)
else:
    for repo in repos:
        print(f"Downloading {repo}...", flush=True)
        snapshot_download(repo_id=repo)
        print(f"[OK] {repo}", flush=True)

print(f"Cache ready: {cache_dir}/hub")
PY
echo "Download complete."
echo "Model cache ready: $CACHE_DIR"