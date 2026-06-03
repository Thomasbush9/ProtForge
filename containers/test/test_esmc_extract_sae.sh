#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --time=01:00:00
#SBATCH --output=/n/home06/tbush/job_logs/%x.%A_%a.out

set -euo pipefail

usage() {
  echo "usage: $0 IMAGE.sif HF_HOME_CACHE_DIR [YAML_FIXTURE] [SIZE] [SAE_TYPE]" >&2
  echo "  YAML_FIXTURE  default: containers/test/fixtures/seq1.yaml" >&2
  echo "  SIZE          6B | 600M | 300M (default: 6B)" >&2
  echo "  SAE_TYPE      all-layers | mlp (default: all-layers)" >&2
  exit 1
}

IMAGE_PATH="${1:-}"
CACHE_DIR="${2:-}"
[[ -n "$IMAGE_PATH" && -n "$CACHE_DIR" ]] || usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_FIXTURE="${3:-${SCRIPT_DIR}/fixtures/seq1.yaml}"
SIZE="${4:-6B}"
SAE_TYPE="${5:-all-layers}"

# SAE layer shards to load (must match files in the HF SAE repo for each size)
case "$SIZE" in
  6B)   LAYERS="30,60" ;;
  600M) LAYERS="18,36" ;;
  300M) LAYERS="12,24" ;;
  *)
    echo "ERROR: no test SAE layers for SIZE=$SIZE" >&2
    exit 1
    ;;
esac

CONTAINER_CACHE="/models/hf"
CONTAINER_YAML="/data/input/$(basename "$YAML_FIXTURE")"
CONTAINER_SCRIPT="/opt/esmc_extract_sae.py"
EXTRACT_RUNNER="${SCRIPT_DIR}/../esmc_extract_sae.py"

if [[ ! -f "$EXTRACT_RUNNER" ]]; then
  echo "ERROR: runner not found: $EXTRACT_RUNNER" >&2
  exit 1
fi
if [[ ! -f "$YAML_FIXTURE" ]]; then
  echo "ERROR: missing fixture: $YAML_FIXTURE" >&2
  exit 1
fi

require_snapshot() {
  local repo="$1"
  local snapshot_dir="$CACHE_DIR/hub/models--${repo//\//--}/snapshots"
  if [[ ! -d "$snapshot_dir" ]] || [[ -z "$(ls -A "$snapshot_dir" 2>/dev/null)" ]]; then
    echo "ERROR: cache missing or empty: $snapshot_dir" >&2
    echo "Run: bash containers/download_scripts/esm_models.sh $CACHE_DIR" >&2
    exit 1
  fi
}

require_snapshot "biohub/ESMC-${SIZE}"
case "$SAE_TYPE" in
  all-layers) require_snapshot "biohub/ESMC-${SIZE}-sae-k64-codebook16384" ;;
  mlp) require_snapshot "biohub/ESMC-${SIZE}-sae-mlp-k64-codebook131072" ;;
  *) echo "ERROR: unknown SAE_TYPE=$SAE_TYPE" >&2; exit 1 ;;
esac

echo "Launching ESMC SAE extract test..."
echo "  image:   $IMAGE_PATH"
echo "  cache:   $CACHE_DIR -> $CONTAINER_CACHE (ro)"
echo "  yaml:    $YAML_FIXTURE"
echo "  size:    $SIZE"
echo "  sae:     $SAE_TYPE"
echo "  layers:  $LAYERS"

singularity exec --nv --cleanenv \
  --env HF_HOME="$CONTAINER_CACHE" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  -B "$CACHE_DIR:$CONTAINER_CACHE:ro" \
  -B "$YAML_FIXTURE:$CONTAINER_YAML:ro" \
  -B "$EXTRACT_RUNNER:$CONTAINER_SCRIPT:ro" \
  "$IMAGE_PATH" \
  python "$CONTAINER_SCRIPT" \
    --cache "$CONTAINER_CACHE" \
    --size "$SIZE" \
    --sae "$SAE_TYPE" \
    --yaml "$CONTAINER_YAML" \
    --layers "$LAYERS"

echo "Run completed."
