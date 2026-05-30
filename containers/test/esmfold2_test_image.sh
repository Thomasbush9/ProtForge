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

SNAPSHOT_DIR="$CACHE_DIR/hub/models--biohub--ESMFold2-Fast/snapshots"
if [[ ! -d "$SNAPSHOT_DIR" ]] || [[ -z "$(ls -A "$SNAPSHOT_DIR" 2>/dev/null)" ]]; then
  echo "ERROR: ESMFold2 cache missing or empty: $SNAPSHOT_DIR" >&2
  echo "Run containers/download_scripts/esm_models.sh with HF_HOME=$CACHE_DIR first." >&2
  exit 1
fi

echo "Launching ESMFold2 test..."
echo "  image:  $IMAGE_PATH"
echo "  cache:  $CACHE_DIR -> $CONTAINER_CACHE (ro)"

singularity exec --nv --cleanenv \
  --env HF_HOME="$CONTAINER_CACHE" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  -B "$CACHE_DIR:$CONTAINER_CACHE:ro" \
  "$IMAGE_PATH" \
  python /opt/run_esmfold2.py --cache "$CONTAINER_CACHE"

echo "Run completed."
