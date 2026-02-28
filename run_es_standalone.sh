#!/bin/bash
set -euo pipefail

# Standalone ES (Evolutionary Scale) Pipeline
# Usage: ./run_es_standalone.sh CIF_DIR [CONFIG_FILE]
# Runs only the ES analysis stage on existing CIF files from Boltz output

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${ROOT_DIR}/slurm_scripts"
CIF_DIR="${1:-}"
CONFIG_FILE="${2:-${ROOT_DIR}/config.yaml}"

if [[ -z "$CIF_DIR" ]]; then
  echo "ERROR: CIF_DIR is required"
  echo ""
  echo "Usage: $0 CIF_DIR [CONFIG_FILE]"
  echo "  CIF_DIR: Directory containing .cif files from Boltz prediction"
  echo "  CONFIG_FILE: Path to YAML configuration file (default: config.yaml in project root)"
  exit 1
fi

if [[ ! -d "$CIF_DIR" ]]; then
  echo "ERROR: CIF_DIR does not exist: $CIF_DIR"
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
ES_SCRIPT_DIR=$(get_config "es.script_dir")
ES_WT_PATH=$(get_config "es.wt_path")
ES_OUTPUT_DIR=$(get_config "es.output_dir")
ES_ENV_PATH=$(get_config "es.env_path")

# SLURM settings
SLURM_LOG_DIR=$(get_config "slurm.log_dir")
SLURM_PARTITION=$(get_config "slurm.partition")
SLURM_ACCOUNT=$(get_config "slurm.account")
SLURM_EMAIL=$(get_config "slurm.email")
SLURM_ES_PARTITION=$(get_config "slurm.es.partition")
[[ -z "$SLURM_ES_PARTITION" ]] && SLURM_ES_PARTITION="$SLURM_PARTITION"

# Validate required parameters
MISSING_PARAMS=()
[[ -z "$ES_SCRIPT_DIR" ]] && MISSING_PARAMS+=("es.script_dir")
[[ -z "$ES_WT_PATH" ]] && MISSING_PARAMS+=("es.wt_path")
[[ -z "$SLURM_LOG_DIR" ]] && MISSING_PARAMS+=("slurm.log_dir")

if ((${#MISSING_PARAMS[@]} > 0)); then
  echo "ERROR: Missing required parameters in config file:"
  for param in "${MISSING_PARAMS[@]}"; do
    echo "  - $param"
  done
  echo ""
  echo "Config file: $CONFIG_FILE"
  exit 1
fi

# Normalize paths
CIF_DIR="$(realpath -m "$CIF_DIR")"
CONFIG_FILE="$(realpath -m "$CONFIG_FILE")"

# Export config-derived env for child scripts (run_es.sh / run_es_array.slrm)
export CONFIG_FILE SLURM_LOG_DIR
export SLURM_PARTITION SLURM_ACCOUNT SLURM_EMAIL SLURM_ES_PARTITION
export ES_SCRIPT_DIR ES_WT_PATH ES_ENV_PREFIX="$ES_ENV_PATH"
[[ -n "$ES_OUTPUT_DIR" ]] && export ES_OUTPUT_DIR

echo "==============================================="
echo "Standalone ES Pipeline"
echo "==============================================="
echo "CIF dir: $CIF_DIR"
echo "Config: $CONFIG_FILE"
echo ""

echo "Launching ES analysis..."
"${SCRIPT_DIR}/run_es.sh" "$CIF_DIR"

echo ""
echo "==============================================="
echo "ES pipeline submitted."
echo "==============================================="
