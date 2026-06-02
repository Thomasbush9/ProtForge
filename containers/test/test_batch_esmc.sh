#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
# Partition/account set by caller via sbatch flags (from config)
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --time=01:00:00
# Log dir from env (caller passes -o)
#SBATCH --output=/n/home06/tbush/job_logs/%x.%A_%a.out

set -euo pipefail

IMAGE_PATH="${1:?usage: $0 IMAGE.sif HF_HOME_CACHE_DIR}"
CACHE_DIR="${2:?usage: $0 IMAGE.sif HF_HOME_CACHE_DIR}"
CONTAINER_CACHE="/models/hf"
CONTAINER_SCRIPT="/opt/run_batch_esmfold2.py"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_RUNNER="${SCRIPT_DIR}/../run_batch_esmfold2.py"
if [[ ! -f "$BATCH_RUNNER" ]]; then
  echo "ERROR: batch runner not found: $BATCH_RUNNER" >&2
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

require_snapshot "biohub/ESMC-6B"

echo "Launching ESMC batch test..."
echo "  image:   $IMAGE_PATH"
echo "  cache:   $CACHE_DIR -> $CONTAINER_CACHE (ro)"
echo "  runner:  $BATCH_RUNNER -> $CONTAINER_SCRIPT (ro)"

singularity exec --nv --cleanenv \
  --env HF_HOME="$CONTAINER_CACHE" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  -B "$CACHE_DIR:$CONTAINER_CACHE:ro" \
  -B "$BATCH_RUNNER:$CONTAINER_SCRIPT:ro" \
  "$IMAGE_PATH" \
  python "$CONTAINER_SCRIPT" --cache "$CONTAINER_CACHE"

echo "Run completed."
