#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48GB
# Partition/account set by caller via sbatch flags (from config)
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --time=01:00:00
# Log dir from env (caller passes -o)
#SBATCH --output=/n/home06/tbush/job_logs/%x.%A_%a.out

set -euo pipefail

usage() {
  echo "usage: $0 IMAGE.sif OPENFOLD_CACHE_DIR [INPUT_YAML_DIR] [OUTPUT_DIR]" >&2
  echo "  INPUT_YAML_DIR  directory of Boltz *.yaml (default: repo fixtures/)" >&2
  echo "  OUTPUT_DIR      writable output root (default: cluster output_tests/openfold3)" >&2
  exit 1
}

IMAGE_PATH="${1:-}"
CACHE_DIR="${2:-}"
[[ -n "$IMAGE_PATH" && -n "$CACHE_DIR" ]] || usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${3:-${SCRIPT_DIR}/fixtures}"
OUTPUT_DIR="${4:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/singularity_dev/images/output_tests/openfold3}"

CONVERTER="${SCRIPT_DIR}/../yaml_to_openfold_json.py"
CONTAINER_DATA="/data"
CONTAINER_CACHE="/models/openfold"
CONTAINER_OUTPUT="/data/output"
WORK_DIR="${OUTPUT_DIR}/.openfold_work"

if [[ ! -f "$CONVERTER" ]]; then
  echo "ERROR: converter not found: $CONVERTER" >&2
  exit 1
fi
if [[ ! -d "$INPUT_DIR" ]] || [[ -z "$(find "$INPUT_DIR" -maxdepth 1 -name '*.yaml' -print -quit)" ]]; then
  echo "ERROR: no *.yaml in INPUT_DIR=$INPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
rm -rf "$WORK_DIR"
python3 "$CONVERTER" \
  --input-dir "$INPUT_DIR" \
  --work-dir "$WORK_DIR" \
  --container-prefix "$CONTAINER_DATA"

RUNNER_ARGS=()
if [[ -f "$WORK_DIR/inference_precomputed.yml" ]]; then
  RUNNER_ARGS=(--runner-yaml "${CONTAINER_DATA}/inference_precomputed.yml")
fi

echo "Launching OpenFold3 test..."
echo "  image:   $IMAGE_PATH"
echo "  cache:   $CACHE_DIR -> $CONTAINER_CACHE (ro)"
echo "  input:   $INPUT_DIR -> query.json via $WORK_DIR"
echo "  output:  $OUTPUT_DIR -> $CONTAINER_OUTPUT (rw)"

singularity exec --nv --cleanenv \
  --env OPENFOLD_CACHE="$CONTAINER_CACHE" \
  -B "$CACHE_DIR:$CONTAINER_CACHE:ro" \
  -B "$WORK_DIR:$CONTAINER_DATA:ro" \
  -B "$OUTPUT_DIR:$CONTAINER_OUTPUT" \
  "$IMAGE_PATH" \
  run_openfold predict \
    --query-json "${CONTAINER_DATA}/query.json" \
    --output-dir "$CONTAINER_OUTPUT" \
    --use-msa-server=False \
    "${RUNNER_ARGS[@]}"

echo "Verifying outputs..."
shopt -s nullglob
found=0
for f in "$OUTPUT_DIR"/*.{cif,pdb}; do
  echo "  ok: $f"
  found=1
done
for sub in "$OUTPUT_DIR"/*/; do
  for f in "$sub"*.{cif,pdb}; do
    echo "  ok: $f"
    found=1
  done
done
if [[ "$found" -eq 0 ]]; then
  echo "ERROR: no .cif or .pdb under $OUTPUT_DIR" >&2
  exit 1
fi

echo "Run completed."
