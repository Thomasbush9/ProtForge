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
  echo "usage: $0 IMAGE.sif OPENFOLD_CACHE_DIR [INPUT_JSON_DIR] [OUTPUT_DIR]" >&2
  echo "  INPUT_JSON_DIR  directory with exactly one OpenFold *.json (default: repo fixtures/)" >&2
  echo "  OUTPUT_DIR      writable output root (default: cluster output_tests/openfold3)" >&2
  exit 1
}

IMAGE_PATH="${1:-}"
CACHE_DIR="${2:-}"
[[ -n "$IMAGE_PATH" && -n "$CACHE_DIR" ]] || usage

INPUT_DIR="${3:-fixtures}"
OUTPUT_DIR="${4:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/singularity_dev/images/output_tests/openfold3}"

CONTAINER_CACHE="/models/openfold"
CONTAINER_OUTPUT="/data/output"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "ERROR: INPUT_DIR is not a directory: $INPUT_DIR" >&2
  exit 1
fi

shopt -s nullglob
json_count="$(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')"
if [[ "$json_count" -ne 1 ]]; then
  echo "ERROR: expected exactly one *.json in INPUT_DIR=$INPUT_DIR, found $json_count" >&2
  exit 1
fi
QUERY_JSON="$(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.json' -print -quit)"
MSA_DIR="${INPUT_DIR}/msa"
if [[ ! -d "$MSA_DIR" ]]; then
  echo "ERROR: expected MSA directory at $MSA_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Launching OpenFold3 test..."
echo "  image:   $IMAGE_PATH"
echo "  cache:   $CACHE_DIR -> $CONTAINER_CACHE (ro)"
echo "  query:   $QUERY_JSON"
echo "  input:   $INPUT_DIR -> $INPUT_DIR (ro)"
echo "  msa:     $MSA_DIR -> $MSA_DIR (ro)"
echo "  output:  $OUTPUT_DIR -> $CONTAINER_OUTPUT (rw)"

singularity exec --nv --cleanenv \
  --env OPENFOLD_CACHE="$CONTAINER_CACHE" \
  -B "$CACHE_DIR:$CONTAINER_CACHE:ro" \
  -B "$INPUT_DIR:$INPUT_DIR:ro" \
  -B "$MSA_DIR:$MSA_DIR:ro" \
  -B "$OUTPUT_DIR:$CONTAINER_OUTPUT" \
  "$IMAGE_PATH" \
  run_openfold predict \
    --query-json "$QUERY_JSON" \
    --output-dir "$CONTAINER_OUTPUT" \
    --use-msa-server=False

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
