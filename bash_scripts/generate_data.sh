#!/bin/bash
set -euo pipefail

# Generate pipeline-ready .fasta folder from a table (TSV/CSV) and reference sequence.
# Input table must contain column 'aaMutations' (e.g. SX123Y:...). Reference: single-sequence .fasta or .yaml.
# Output: directory of .fasta files. Set input.fasta_dir in config.yaml to this path.
#
# Usage: ./generate_data.sh --data /path/to/file.tsv --original /path/to/reference.fasta [OPTIONS]
#
# Required:
#   --data PATH       Table with aaMutations column (.tsv or .csv)
#   --original PATH   Reference sequence (.fasta or .yaml)
# Optional:
#   --msa PATH        MSA file path (stored in fasta header)
#   --output_dir DIR  Output base dir (default: $MAIN_DIR/data/generated)
#   --file_type       cluster|fasta|yaml (default: cluster for pipeline .fasta)
#   --subsample N     After generating, create subsample of N sequences
#   --subsample_mode  balanced|fixed (default: balanced)
#   --num_mut         For fixed mode: exact number of mutations per sequence
#   --seed            RNG seed for subsampling (default: 42)
#   --venv DIR        Use this Python venv (create + install deps if missing). Default: $MAIN_DIR/.venv_data

MAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MAIN_DIR" || exit 1

DATA_PATH=""
ORIGINAL_PATH=""
MSA_PATH=""
OUTPUT_DIR="${MAIN_DIR}/data/generated"
FILE_TYPE="cluster"
SUBSAMPLE_N=""
SUBSAMPLE_MODE="balanced"
NUM_MUT=""
SEED="42"
VENV_DIR="${MAIN_DIR}/.venv_data"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)          DATA_PATH="$2"; shift 2 ;;
    --original)      ORIGINAL_PATH="$2"; shift 2 ;;
    --msa)           MSA_PATH="$2"; shift 2 ;;
    --output_dir)    OUTPUT_DIR="$2"; shift 2 ;;
    --file_type)     FILE_TYPE="$2"; shift 2 ;;
    --subsample)     SUBSAMPLE_N="$2"; shift 2 ;;
    --subsample_mode) SUBSAMPLE_MODE="$2"; shift 2 ;;
    --num_mut)       NUM_MUT="$2"; shift 2 ;;
    --seed)          SEED="$2"; shift 2 ;;
    --venv)          VENV_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Ensure Python venv exists and activate it (for pandas, pyyaml, tqdm, numpy)
ensure_venv() {
  local venv="$1"
  if [[ -d "$venv" && -f "$venv/bin/activate" ]]; then
    return 0
  fi
  echo "Creating venv at $venv and installing dependencies..."
  python3 -m venv "$venv" || { echo "ERROR: python3 -m venv failed"; exit 1; }
  "$venv/bin/pip" install -q -r "$MAIN_DIR/requirements-data.txt" || { echo "ERROR: pip install failed"; exit 1; }
}
ensure_venv "$VENV_DIR"
PYTHON_CMD="$VENV_DIR/bin/python"

[[ -z "$DATA_PATH" ]] && { echo "ERROR: --data required"; exit 1; }
[[ -z "$ORIGINAL_PATH" ]] && { echo "ERROR: --original required"; exit 1; }
[[ ! -f "$DATA_PATH" ]] && { echo "ERROR: Data file not found: $DATA_PATH"; exit 1; }
[[ ! -f "$ORIGINAL_PATH" ]] && { echo "ERROR: Original/reference file not found: $ORIGINAL_PATH"; exit 1; }

# Infer separator for Python
if [[ "$DATA_PATH" == *.csv ]] || [[ "$DATA_PATH" == *.CSV ]]; then
  SEP=","
else
  SEP=$'\t'
fi

mkdir -p "$OUTPUT_DIR"
TRAINING_DIR="${OUTPUT_DIR}/training_data"

echo "Running generate_data.py..."
"$PYTHON_CMD" -m utils.generate_data \
  --data "$(realpath -m "$DATA_PATH")" \
  --original "$(realpath -m "$ORIGINAL_PATH")" \
  --file_type "$FILE_TYPE" \
  --output_dir "$(realpath -m "$OUTPUT_DIR")" \
  --sep "$SEP" \
  ${MSA_PATH:+--msa "$(realpath -m "$MSA_PATH")"}

if [[ -n "$SUBSAMPLE_N" ]]; then
  echo "Running generate_subsamples.py..."
  SUBSAMPLE_OUTPUT=$("$PYTHON_CMD" -m utils.generate_subsamples \
    --dataset "$(realpath -m "$DATA_PATH")" \
    --n "$SUBSAMPLE_N" \
    --main_dir "$(realpath -m "$TRAINING_DIR")" \
    --mode "$SUBSAMPLE_MODE" \
    --seed "$SEED" \
    --sep "$SEP" \
    $([[ -n "$NUM_MUT" ]] && echo "--num_mut $NUM_MUT"))
  echo "$SUBSAMPLE_OUTPUT"
  FINAL_DIR=$(echo "$SUBSAMPLE_OUTPUT" | grep -E '^SUBSAMPLE_OUTPUT_DIR=' | sed 's/^SUBSAMPLE_OUTPUT_DIR=//')
  [[ -z "$FINAL_DIR" ]] && { echo "ERROR: Could not get SUBSAMPLE_OUTPUT_DIR from generate_subsamples"; exit 1; }
else
  FINAL_DIR="$(realpath -m "$TRAINING_DIR")"
fi

echo ""
echo "Done. Set input.fasta_dir in config.yaml to:"
echo "  $FINAL_DIR"
