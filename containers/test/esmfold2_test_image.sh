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

usage() {
  echo "usage: $0 IMAGE.sif HF_HOME_CACHE_DIR [INPUT_YAML_DIR] [OUTPUT_DIR]" >&2
  echo "  INPUT_YAML_DIR  directory of Boltz *.yaml (default: repo fixtures/)" >&2
  echo "  OUTPUT_DIR      writable output root (default: cluster output_tests/esmfold_batch)" >&2
  exit 1
}

IMAGE_PATH="${1:-}"
CACHE_DIR="${2:-}"
[[ -n "$IMAGE_PATH" && -n "$CACHE_DIR" ]] || usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${3:-${SCRIPT_DIR}/fixtures}"
OUTPUT_DIR="${4:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/singularity_dev/images/output_tests/esmfold_batch}"

CONTAINER_CACHE="/models/hf"
CONTAINER_INPUT="/data/input"
CONTAINER_OUTPUT="/data/output"
CONTAINER_SCRIPT="/opt/run_batch_esmfold.py"
VARIANT="fast"

BATCH_RUNNER="${SCRIPT_DIR}/../run_batch_esmfold.py"
if [[ ! -f "$BATCH_RUNNER" ]]; then
  echo "ERROR: batch runner not found: $BATCH_RUNNER" >&2
  exit 1
fi
if [[ ! -d "$INPUT_DIR" ]] || [[ -z "$(find "$INPUT_DIR" -maxdepth 1 -name '*.yaml' -print -quit)" ]]; then
  echo "ERROR: no *.yaml in INPUT_DIR=$INPUT_DIR" >&2
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

require_snapshot "biohub/ESMFold2-Fast"

mkdir -p "$OUTPUT_DIR"

echo "Launching ESMFold2 batch test..."
echo "  image:      $IMAGE_PATH"
echo "  cache:      $CACHE_DIR -> $CONTAINER_CACHE (ro)"
echo "  input:      $INPUT_DIR -> $CONTAINER_INPUT (ro)"
echo "  output:     $OUTPUT_DIR -> $CONTAINER_OUTPUT (rw)"
echo "  variant:    $VARIANT"
echo "  runner:     $BATCH_RUNNER -> $CONTAINER_SCRIPT (ro)"

singularity exec --nv --cleanenv \
  --env HF_HOME="$CONTAINER_CACHE" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  -B "$CACHE_DIR:$CONTAINER_CACHE:ro" \
  -B "$INPUT_DIR:$CONTAINER_INPUT:ro" \
  -B "$OUTPUT_DIR:$CONTAINER_OUTPUT" \
  -B "$BATCH_RUNNER:$CONTAINER_SCRIPT:ro" \
  "$IMAGE_PATH" \
  python "$CONTAINER_SCRIPT" \
    --cache "$CONTAINER_CACHE" \
    --input-dir "$CONTAINER_INPUT" \
    --output-dir "$CONTAINER_OUTPUT"

echo "Verifying outputs..."
shopt -s nullglob
for yaml in "$INPUT_DIR"/*.yaml; do
  stem="$(basename "$yaml" .yaml)"
  base="$OUTPUT_DIR/$stem/$VARIANT"
  for f in structure.cif plddt.npy metrics.pt; do
    if [[ ! -f "$base/$f" ]]; then
      echo "ERROR: missing $base/$f" >&2
      exit 1
    fi
    echo "  ok: $base/$f"
  done
done

echo "Run completed."
