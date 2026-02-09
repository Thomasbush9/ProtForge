#!/bin/bash
set -euo pipefail

# Usage: ./run.sh [CONFIG_FILE]
# Example: ./run.sh pipeline_config.yaml
#          ./run.sh  (uses pipeline_config.yaml in script directory)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm_scripts/"
CONFIG_FILE="${1:-${SCRIPT_DIR}/pipeline_config.yaml}"

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: Config file not found: $CONFIG_FILE"
  echo ""
  echo "Usage: $0 [CONFIG_FILE]"
  echo "  CONFIG_FILE: Path to YAML configuration file (default: pipeline_config.yaml)"
  echo ""
  echo "Example config file: ${SCRIPT_DIR}/pipeline_config.example.yaml"
  exit 1
fi

PARSE_SCRIPT="${SCRIPT_DIR}/parse_config.py"

# Function to get config value
get_config() {
  local key="$1"
  python3 "$PARSE_SCRIPT" "$CONFIG_FILE" "$key" 2>/dev/null || echo ""
}

# Read configuration values
INPUT_DIR=$(get_config "input.fasta_dir")
OUTPUT_PARENT_DIR=$(get_config "output.parent_dir")
MSA_MAX_FILES_PER_JOB=$(get_config "msa.max_files_per_job")
MSA_ARRAY_MAX_CONCURRENCY=$(get_config "msa.array_max_concurrency")
MAX_FILES_PER_JOB=$(get_config "boltz.max_files_per_job")
BOLTZ_ARRAY_MAX_CONCURRENCY=$(get_config "boltz.array_max_concurrency")
BOLTZ_RECYCLING_STEPS=$(get_config "boltz.recycling_steps")
BOLTZ_DIFFUSION_SAMPLES=$(get_config "boltz.diffusion_samples")
ESM_N=$(get_config "esm.num_chunks")
ESM_ARRAY_MAX_CONCURRENCY=$(get_config "esm.array_max_concurrency")
ES_SCRIPT_DIR=$(get_config "es.script_dir")
ES_WT_PATH=$(get_config "es.wt_path")
ES_ARRAY_MAX_CONCURRENCY=$(get_config "es.array_max_concurrency")

# Set defaults
ARRAY_MAX_CONCURRENCY="${MSA_ARRAY_MAX_CONCURRENCY:-100}"
BOLTZ_ARRAY_MAX_CONCURRENCY="${BOLTZ_ARRAY_MAX_CONCURRENCY:-${ARRAY_MAX_CONCURRENCY}}"
ESM_ARRAY_MAX_CONCURRENCY="${ESM_ARRAY_MAX_CONCURRENCY:-${ARRAY_MAX_CONCURRENCY}}"
ES_ARRAY_MAX_CONCURRENCY="${ES_ARRAY_MAX_CONCURRENCY:-10}"

# Validate required parameters
MISSING_PARAMS=()
[[ -z "$INPUT_DIR" ]] && MISSING_PARAMS+=("input.fasta_dir")
[[ -z "$OUTPUT_PARENT_DIR" ]] && MISSING_PARAMS+=("output.parent_dir")
[[ -z "$MSA_MAX_FILES_PER_JOB" ]] && MISSING_PARAMS+=("msa.max_files_per_job")
[[ -z "$MAX_FILES_PER_JOB" ]] && MISSING_PARAMS+=("boltz.max_files_per_job")
[[ -z "$ESM_N" ]] && MISSING_PARAMS+=("esm.num_chunks")

if ((${#MISSING_PARAMS[@]} > 0)); then
  echo "ERROR: Missing required parameters in config file:"
  for param in "${MISSING_PARAMS[@]}"; do
    echo "  - $param"
  done
  echo ""
  echo "Config file: $CONFIG_FILE"
  echo "Example config: ${SCRIPT_DIR}/pipeline_config.example.yaml"
  exit 1
fi

MSA_SCRIPT="${SCRIPT_DIR}/split_and_run_msa.sh"
POST_PROCESS_SCRIPT="${SCRIPT_DIR}/post_process_msa.slrm"
BOLTZ_SCRIPT="${SCRIPT_DIR}/split_and_run_boltz.sh"
ESM_SCRIPT="${SCRIPT_DIR}/run_esm.sh"

# Normalize paths
INPUT_DIR="$(realpath -m "$INPUT_DIR")"
OUTPUT_PARENT_DIR="$(realpath -m "$OUTPUT_PARENT_DIR")"

echo "==============================================="
echo "Orchestrating MSA -> YAML conversion -> ESM + Boltz (parallel)"
echo "==============================================="
echo "Input dir: $INPUT_DIR"
echo "Output parent dir: $OUTPUT_PARENT_DIR"
echo ""
