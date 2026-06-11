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
  echo "usage: $0 IMAGE.sif HF_HOME_CACHE_DIR [INPUT_DIR] [OUTPUT_DIR] [SIZE] [MODE]" >&2
  echo "  INPUT_DIR  directory of *.yaml or *.fasta/*.fa (default: repo fixtures/)" >&2
  echo "  OUTPUT_DIR writable output root (default: cluster output_tests/esmc_batch)" >&2
  echo "  SIZE       6B | 600M | 300M | all (default: 6B)" >&2
  echo "  MODE       embeddings | logits (default: embeddings)" >&2
  echo "             logits adds --save-logits (writes logits.npy for mutation scans)" >&2
  exit 1
}

IMAGE_PATH="${1:-}"
CACHE_DIR="${2:-}"
[[ -n "$IMAGE_PATH" && -n "$CACHE_DIR" ]] || usage
#TODO: bind script to container
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${3:-${SCRIPT_DIR}/fixtures}"
OUTPUT_DIR="${4:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/singularity_dev/images/output_tests/esmc_batch}"
SIZE="${5:-6B}"
MODE="${6:-embeddings}"
[[ "$MODE" == "embeddings" || "$MODE" == "logits" ]] || usage

CONTAINER_CACHE="/models/hf"
CONTAINER_INPUT="/data/input"
CONTAINER_OUTPUT="/data/output"
CONTAINER_SCRIPT="/opt/run_batch_esmc.py"

BATCH_RUNNER="${SCRIPT_DIR}/../run_batch_esmc.py"
if [[ ! -f "$BATCH_RUNNER" ]]; then
  echo "ERROR: batch runner not found: $BATCH_RUNNER" >&2
  exit 1
fi
if [[ ! -d "$INPUT_DIR" ]] || [[ -z "$(find "$INPUT_DIR" \( -name '*.yaml' -o -name '*.fasta' -o -name '*.fa' \) -print -quit)" ]]; then
  echo "ERROR: no *.yaml/*.fasta/*.fa in INPUT_DIR=$INPUT_DIR" >&2
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
if [[ "$SIZE" == "600M" || "$SIZE" == "all" ]]; then
  require_snapshot "biohub/ESMC-600M"
fi
if [[ "$SIZE" == "300M" || "$SIZE" == "all" ]]; then
  require_snapshot "biohub/ESMC-300M"
fi

mkdir -p "$OUTPUT_DIR"

# Mode -> extra runner flag + the artifact each size dir should contain.
if [[ "$MODE" == "logits" ]]; then
  EXTRA_ARGS=(--save-logits)
  EXPECTED_FILE="logits.npy"
else
  EXTRA_ARGS=()
  EXPECTED_FILE="outputs.pt"
fi

echo "Launching ESMC batch test..."
echo "  image:      $IMAGE_PATH"
echo "  cache:      $CACHE_DIR -> $CONTAINER_CACHE (ro)"
echo "  input:      $INPUT_DIR -> $CONTAINER_INPUT (ro)"
echo "  output:     $OUTPUT_DIR -> $CONTAINER_OUTPUT (rw)"
echo "  size:       $SIZE"
echo "  mode:       $MODE (expecting $EXPECTED_FILE)"
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
  --output-dir "$CONTAINER_OUTPUT" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  --size "$SIZE"
#TODO: verificatoin will be moved to snakemake rules
echo "Verifying outputs..."
if [[ "$SIZE" == "all" ]]; then
  SIZES=(6B 600M 300M)
else
  SIZES=("$SIZE")
fi

shopt -s nullglob
for input in "$INPUT_DIR"/*.yaml "$INPUT_DIR"/*.fasta "$INPUT_DIR"/*.fa; do
  stem="$(basename "$input")"
  stem="${stem%.*}"
  for s in "${SIZES[@]}"; do
    out_file="$OUTPUT_DIR/$stem/$s/$EXPECTED_FILE"
    if [[ ! -f "$out_file" ]]; then
      echo "ERROR: missing $out_file" >&2
      exit 1
    fi
    echo "  ok: $out_file"
  done
done

echo "Run completed."
