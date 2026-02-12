#!/bin/bash
set -euo pipefail

# Standalone Boltz Pipeline
# Usage: ./run_boltz.sh YAML_DIR [CONFIG_FILE]
# Runs only the Boltz structure prediction stage on existing YAML files

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${ROOT_DIR}/slurm_scripts"
YAML_DIR="${1:-}"
CONFIG_FILE="${2:-${ROOT_DIR}/config.yaml}"

if [[ -z "$YAML_DIR" ]]; then
  echo "ERROR: YAML_DIR is required"
  echo ""
  echo "Usage: $0 YAML_DIR [CONFIG_FILE]"
  echo "  YAML_DIR: Directory containing .yaml files for Boltz prediction"
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
MAX_FILES_PER_JOB=$(get_config "boltz.max_files_per_job")
BOLTZ_ARRAY_MAX_CONCURRENCY=$(get_config "boltz.array_max_concurrency")
BOLTZ_RECYCLING_STEPS=$(get_config "boltz.recycling_steps")
BOLTZ_DIFFUSION_SAMPLES=$(get_config "boltz.diffusion_samples")

# SLURM settings
SLURM_LOG_DIR=$(get_config "slurm.log_dir")
SLURM_PARTITION=$(get_config "slurm.partition")
SLURM_ACCOUNT=$(get_config "slurm.account")
SLURM_BOLTZ_PARTITION=$(get_config "slurm.boltz.partition")
[[ -z "$SLURM_BOLTZ_PARTITION" ]] && SLURM_BOLTZ_PARTITION="$SLURM_PARTITION"
SLURM_CHECKER_BOLTZ_PARTITION=$(get_config "slurm.checker_boltz.partition")
[[ -z "$SLURM_CHECKER_BOLTZ_PARTITION" ]] && SLURM_CHECKER_BOLTZ_PARTITION="$SLURM_BOLTZ_PARTITION"

# Boltz-specific paths
BOLTZ_CACHE=$(get_config "boltz.cache_dir")
BOLTZ_COLABFOLD_DB=$(get_config "boltz.colabfold_db")
BOLTZ_ENV_PATH=$(get_config "boltz.env_path")
COLABFOLD_BIN=$(get_config "msa.colabfold_bin")

# Set defaults
BOLTZ_ARRAY_MAX_CONCURRENCY="${BOLTZ_ARRAY_MAX_CONCURRENCY:-100}"
BOLTZ_RECYCLING_STEPS="${BOLTZ_RECYCLING_STEPS:-8}"
BOLTZ_DIFFUSION_SAMPLES="${BOLTZ_DIFFUSION_SAMPLES:-10}"

# Validate required parameters
MISSING_PARAMS=()
[[ -z "$MAX_FILES_PER_JOB" ]] && MISSING_PARAMS+=("boltz.max_files_per_job")
[[ -z "$SLURM_LOG_DIR" ]] && MISSING_PARAMS+=("slurm.log_dir")
[[ -z "$BOLTZ_CACHE" ]] && MISSING_PARAMS+=("boltz.cache_dir")
[[ -z "$BOLTZ_ENV_PATH" ]] && MISSING_PARAMS+=("boltz.env_path")

if ((${#MISSING_PARAMS[@]} > 0)); then
  echo "ERROR: Missing required parameters in config file:"
  for param in "${MISSING_PARAMS[@]}"; do
    echo "  - $param"
  done
  echo ""
  echo "Config file: $CONFIG_FILE"
  exit 1
fi

BOLTZ_SCRIPT="${SCRIPT_DIR}/split_and_run_boltz.sh"

# Normalize paths
YAML_DIR="$(realpath -m "$YAML_DIR")"
CONFIG_FILE="$(realpath -m "$CONFIG_FILE")"

# Export config-derived env for child scripts
export CONFIG_FILE SLURM_LOG_DIR
export SLURM_PARTITION SLURM_ACCOUNT SLURM_BOLTZ_PARTITION SLURM_CHECKER_BOLTZ_PARTITION
export BOLTZ_CACHE BOLTZ_COLABFOLD_DB BOLTZ_ENV_PATH
[[ -n "$COLABFOLD_BIN" ]] && export PATH="${COLABFOLD_BIN}:${PATH}"

echo "==============================================="
echo "Standalone Boltz Pipeline"
echo "==============================================="
echo "YAML dir: $YAML_DIR"
echo "Config: $CONFIG_FILE"
echo ""

echo "Launching Boltz prediction..."
boltz_output=$("$BOLTZ_SCRIPT" "$YAML_DIR" "$MAX_FILES_PER_JOB" "$BOLTZ_ARRAY_MAX_CONCURRENCY" "$BOLTZ_RECYCLING_STEPS" "$BOLTZ_DIFFUSION_SAMPLES")
echo "$boltz_output"

# Extract job ID from output
BOLTZ_JOB_ID=$(echo "$boltz_output" | grep -oP 'Submitted array job \K[0-9]+' | head -1 || echo "")

if [[ -z "$BOLTZ_JOB_ID" ]]; then
  echo "WARNING: Could not extract Boltz job ID from output"
else
  echo ""
  echo "Boltz array job ID: ${BOLTZ_JOB_ID}"

  # Submit checker job
  if [[ -f "${SCRIPT_DIR}/run_checker_boltz.slrm" ]]; then
    CHECKER_BOLTZ_JOB_ID=$(sbatch --parsable --dependency=afternotok:"$BOLTZ_JOB_ID" \
      ${SLURM_CHECKER_BOLTZ_PARTITION:+-p "$SLURM_CHECKER_BOLTZ_PARTITION"} ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} \
      -o "${SLURM_LOG_DIR}/%x.%j.out" \
      --export=ALL,OUTPUT_DIR="$YAML_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
      "${SCRIPT_DIR}/run_checker_boltz.slrm" 2>/dev/null || echo "")
    [[ -n "$CHECKER_BOLTZ_JOB_ID" ]] && echo "Boltz checker job ID: ${CHECKER_BOLTZ_JOB_ID} (will run if Boltz job fails)"
  fi
fi

echo ""
echo "==============================================="
echo "Boltz pipeline submitted."
echo "To check for errors and retry: ./slurm_scripts/checker.sh boltz $YAML_DIR $CONFIG_FILE"
echo "==============================================="
