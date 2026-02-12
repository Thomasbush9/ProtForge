#!/bin/bash
set -euo pipefail

# Generate pipeline-ready files from a CSV/TSV table.
#
# Two input modes:
#   1. MUTATIONS MODE: CSV with 'aaMutations' column + reference sequence
#   2. SEQUENCES MODE: CSV with 'name' and 'sequence' columns (no reference needed)
#
# Output: directory of .fasta or .yaml files. Set input.fasta_dir (or input.yaml_dir) in config.yaml.
#
# Usage:
#   Mutations mode:
#     ./generate_data.sh --data mutations.tsv --original reference.fasta [OPTIONS]
#
#   Sequences mode:
#     ./generate_data.sh --data sequences.csv [OPTIONS]
#
# Required:
#   --data PATH       CSV/TSV file with either:
#                     - 'aaMutations' column (mutations mode)
#                     - 'name' and 'sequence' columns (sequences mode)
#
# For mutations mode only:
#   --original PATH   Reference sequence (.fasta or .yaml)
#
# Optional:
#   --mode            Force mode: mutations|sequences (auto-detected if not specified)
#   --msa PATH        MSA file path (stored in output files)
#   --output_dir DIR  Output base dir (default: $MAIN_DIR/data/generated)
#   --file_type       cluster|fasta|yaml (default: cluster)
#                     - cluster/fasta: outputs .fasta for MSA pipeline
#                     - yaml: outputs .yaml to skip MSA (use with Boltz directly)
#   --subsample N     After generating, create subsample of N sequences (mutations mode only)
#   --subsample_mode  balanced|fixed (default: balanced)
#   --num_mut         For fixed mode: exact number of mutations per sequence
#   --seed            RNG seed for subsampling (default: 42)
#   --venv DIR        Use this Python venv (create + install deps if missing)

MAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MAIN_DIR" || exit 1

DATA_PATH=""
ORIGINAL_PATH=""
MSA_PATH=""
OUTPUT_DIR="${MAIN_DIR}/data/generated"
FILE_TYPE="cluster"
INPUT_MODE=""
SUBSAMPLE_N=""
SUBSAMPLE_MODE="balanced"
NUM_MUT=""
SEED="42"
VENV_DIR="${MAIN_DIR}/.venv_data"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)          DATA_PATH="$2"; shift 2 ;;
    --original)      ORIGINAL_PATH="$2"; shift 2 ;;
    --mode)          INPUT_MODE="$2"; shift 2 ;;
    --msa)           MSA_PATH="$2"; shift 2 ;;
    --output_dir)    OUTPUT_DIR="$2"; shift 2 ;;
    --file_type)     FILE_TYPE="$2"; shift 2 ;;
    --subsample)     SUBSAMPLE_N="$2"; shift 2 ;;
    --subsample_mode) SUBSAMPLE_MODE="$2"; shift 2 ;;
    --num_mut)       NUM_MUT="$2"; shift 2 ;;
    --seed)          SEED="$2"; shift 2 ;;
    --venv)          VENV_DIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --data <file.csv> [--original <ref.fasta>] [OPTIONS]"
      echo ""
      echo "Modes:"
      echo "  Sequences: CSV with 'name' and 'sequence' columns"
      echo "  Mutations: CSV with 'aaMutations' column (requires --original)"
      echo ""
      echo "Options:"
      echo "  --mode mutations|sequences  Force input mode (auto-detected)"
      echo "  --file_type cluster|fasta|yaml  Output format (default: cluster)"
      echo "  --output_dir DIR  Output directory"
      echo "  --msa PATH  MSA file path to embed in output"
      exit 0
      ;;
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
[[ ! -f "$DATA_PATH" ]] && { echo "ERROR: Data file not found: $DATA_PATH"; exit 1; }

# Validate original path only if provided or in mutations mode
if [[ -n "$ORIGINAL_PATH" ]] && [[ ! -f "$ORIGINAL_PATH" ]]; then
  echo "ERROR: Original/reference file not found: $ORIGINAL_PATH"
  exit 1
fi

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
  --file_type "$FILE_TYPE" \
  --output_dir "$(realpath -m "$OUTPUT_DIR")" \
  --sep "$SEP" \
  ${ORIGINAL_PATH:+--original "$(realpath -m "$ORIGINAL_PATH")"} \
  ${INPUT_MODE:+--mode "$INPUT_MODE"} \
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
