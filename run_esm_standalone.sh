#!/bin/bash
set -euo pipefail

# Standalone ESM Pipeline
# Usage: ./run_esm_standalone.sh YAML_DIR [CONFIG_FILE]
# Runs only the ESM embedding generation stage on existing YAML files

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${ROOT_DIR}/slurm_scripts"
YAML_DIR="${1:-}"
CONFIG_FILE="${2:-${ROOT_DIR}/config.yaml}"

if [[ -z "$YAML_DIR" ]]; then
  echo "ERROR: YAML_DIR is required"
  echo ""
  echo "Usage: $0 YAML_DIR [CONFIG_FILE]"
  echo "  YAML_DIR: Directory containing .yaml files for ESM embedding generation"
  echo "  CONFIG_FILE: Path to YAML configuration file (default: config.yaml in project root)"
  exit 1
fi

if [[ ! -d "$YAML_DIR" ]]; then
  echo "ERROR: YAML_DIR does not exist: $YAML_DIR"
  exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: Config file not found: $CONFIG_FILE"
  exit 1
fi

PARSE_SCRIPT="${SCRIPT_DIR}/parse_config.py"

get_config() {
  local key="$1"
  python3 "$PARSE_SCRIPT" "$CONFIG_FILE" "$key" 2>/dev/null || echo ""
}

# Read configuration values
OUTPUT_PARENT_DIR=$(get_config "output.parent_dir")
ESM_N=$(get_config "esm.num_chunks")
ESM_ARRAY_MAX_CONCURRENCY=$(get_config "esm.array_max_concurrency")

# SLURM settings
SLURM_LOG_DIR=$(get_config "slurm.log_dir")
SLURM_PARTITION=$(get_config "slurm.partition")
SLURM_ACCOUNT=$(get_config "slurm.account")
SLURM_EMAIL=$(get_config "slurm.email")
SLURM_ESM_PARTITION=$(get_config "slurm.esm.partition")
[[ -z "$SLURM_ESM_PARTITION" ]] && SLURM_ESM_PARTITION="$SLURM_PARTITION"

# ESM-specific paths
ESM_ENV_PATH=$(get_config "esm.env_path")
ESM_WORK_DIR=$(get_config "esm.work_dir")
ESM_CACHE_DIR=$(get_config "esm.cache_dir")

# Set defaults
ESM_N="${ESM_N:-1}"
ESM_ARRAY_MAX_CONCURRENCY="${ESM_ARRAY_MAX_CONCURRENCY:-20}"

# Validate required parameters
MISSING_PARAMS=()
[[ -z "$OUTPUT_PARENT_DIR" ]] && MISSING_PARAMS+=("output.parent_dir")
[[ -z "$ESM_N" ]] && MISSING_PARAMS+=("esm.num_chunks")
[[ -z "$SLURM_LOG_DIR" ]] && MISSING_PARAMS+=("slurm.log_dir")
[[ -z "$ESM_ENV_PATH" ]] && MISSING_PARAMS+=("esm.env_path")

if ((${#MISSING_PARAMS[@]} > 0)); then
  echo "ERROR: Missing required parameters in config file:"
  for param in "${MISSING_PARAMS[@]}"; do
    echo "  - $param"
  done
  echo ""
  echo "Config file: $CONFIG_FILE"
  exit 1
fi

ESM_SCRIPT="${SCRIPT_DIR}/run_esm.sh"

# Normalize paths
YAML_DIR="$(realpath -m "$YAML_DIR")"
OUTPUT_PARENT_DIR="$(realpath -m "$OUTPUT_PARENT_DIR")"
CONFIG_FILE="$(realpath -m "$CONFIG_FILE")"

# Export config-derived env for child scripts
export CONFIG_FILE SLURM_LOG_DIR
export SLURM_PARTITION SLURM_ACCOUNT SLURM_EMAIL SLURM_ESM_PARTITION
export ESM_ENV_PREFIX="$ESM_ENV_PATH" ESM_WORK_DIR ESM_CACHE_DIR

echo "==============================================="
echo "Standalone ESM Pipeline"
echo "==============================================="
echo "YAML dir: $YAML_DIR"
echo "Output parent dir: $OUTPUT_PARENT_DIR"
echo "Config: $CONFIG_FILE"
echo ""

echo "Launching ESM embedding generation..."
esm_output=$("$ESM_SCRIPT" "$YAML_DIR" "$ESM_N" "$OUTPUT_PARENT_DIR" "$ESM_ARRAY_MAX_CONCURRENCY")
echo "$esm_output"

# Extract job ID from output
ESM_JOB_ID=$(echo "$esm_output" | grep -oP 'Submitted array job \K[0-9]+' | head -1 || echo "")

if [[ -z "$ESM_JOB_ID" ]]; then
  echo "WARNING: Could not extract ESM job ID from output"
else
  echo ""
  echo "ESM array job ID: ${ESM_JOB_ID}"

  # Submit checker job
  CHECKER_ESM_SCRIPT="${SCRIPT_DIR}/checker_esm.sh"
  if [[ -f "$CHECKER_ESM_SCRIPT" ]]; then
    CHECKER_ESM_WRAPPER="${SCRIPT_DIR}/run_checker_esm.slrm"
    if [[ -f "$CHECKER_ESM_WRAPPER" ]]; then
      CHECKER_ESM_JOB_ID=$(sbatch --parsable --dependency=afternotok:"$ESM_JOB_ID" \
        ${SLURM_ESM_PARTITION:+-p "$SLURM_ESM_PARTITION"} ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} ${SLURM_EMAIL:+--mail-type=ALL --mail-user="$SLURM_EMAIL"} \
        -o "${SLURM_LOG_DIR}/%x.%j.out" \
        --export=ALL,OUTPUT_DIR="$YAML_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
        "$CHECKER_ESM_WRAPPER" 2>/dev/null || echo "")
      [[ -n "$CHECKER_ESM_JOB_ID" ]] && echo "ESM checker job ID: ${CHECKER_ESM_JOB_ID} (will run if ESM job fails)"
    else
      # Fallback: submit checker directly
      CHECKER_ESM_JOB_ID=$(sbatch --parsable --dependency=afternotok:"$ESM_JOB_ID" \
        ${SLURM_ESM_PARTITION:+-p "$SLURM_ESM_PARTITION"} ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} ${SLURM_EMAIL:+--mail-type=ALL --mail-user="$SLURM_EMAIL"} \
        -o "${SLURM_LOG_DIR}/%x.%j.out" \
        --export=ALL,OUTPUT_DIR="$YAML_DIR" \
        --wrap="bash $CHECKER_ESM_SCRIPT $YAML_DIR" 2>/dev/null || echo "")
      [[ -n "$CHECKER_ESM_JOB_ID" ]] && echo "ESM checker job ID: ${CHECKER_ESM_JOB_ID} (will run if ESM job fails)"
    fi
  fi
fi

echo ""
echo "==============================================="
echo "ESM pipeline submitted."
echo "To check for errors and retry: ./slurm_scripts/checker.sh esm $YAML_DIR $CONFIG_FILE"
echo "==============================================="
