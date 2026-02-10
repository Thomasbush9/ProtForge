#!/bin/bash
set -euo pipefail

# Usage: ./run_es.sh ES_ROOT_DIR
# ES_ROOT_DIR: Directory containing .cif files to analyze (e.g. pipeline output with Boltz predictions)
# Other params (ES_SCRIPT_DIR, ES_WT_PATH, ES_OUTPUT_DIR, ES_ENV_PREFIX, SLURM_LOG_DIR) from env (run.sh exports from config)

ES_ROOT_DIR="${1:-}"
if [[ -z "$ES_ROOT_DIR" ]]; then
  echo "Usage: $0 ES_ROOT_DIR"
  echo "  ES_ROOT_DIR: Directory containing .cif files (e.g. pipeline output)"
  exit 1
fi
ES_ROOT_DIR="$(realpath -m "$ES_ROOT_DIR")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ES_SCRIPT="${SCRIPT_DIR}/run_es_array.slrm"
if [[ ! -f "$ES_SCRIPT" ]]; then
  echo "ERROR: run_es_array.slrm not found at $ES_SCRIPT"
  exit 1
fi

SLURM_OUTPUT="${SLURM_LOG_DIR:-/tmp}/%x.%j.out"
SBATCH_OPTS=()
[[ -n "${SLURM_ES_PARTITION:-}" ]] && SBATCH_OPTS+=(-p "$SLURM_ES_PARTITION")
[[ -n "${SLURM_ACCOUNT:-}" ]] && SBATCH_OPTS+=(--account "$SLURM_ACCOUNT")
echo "Submitting ES analysis job (root dir: $ES_ROOT_DIR)..."
sbatch "${SBATCH_OPTS[@]}" -o "$SLURM_OUTPUT" \
  --export=ALL,ES_ROOT_DIR="$ES_ROOT_DIR" \
  "$ES_SCRIPT"
echo "ES job submitted."
