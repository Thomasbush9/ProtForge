#!/bin/bash
set -euo pipefail

# Usage: ./checker.sh TYPE ROOT_DIR [CONFIG_FILE]
# TYPE: msa | boltz | esm
# Dispatches to checker_msa.sh, checker_boltz.sh, or checker_esm.sh with config-derived env.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSE_SCRIPT="${SCRIPT_DIR}/parse_config.py"

get_config() {
  local key="$1"
  [[ -n "${CONFIG_FILE:-}" && -f "$CONFIG_FILE" ]] && python3 "$PARSE_SCRIPT" "$CONFIG_FILE" "$key" 2>/dev/null || echo ""
}

TYPE="${1:-}"
ROOT_DIR="${2:-}"
CONFIG_FILE="${3:-}"

if [[ -z "$TYPE" || -z "$ROOT_DIR" ]]; then
  echo "Usage: $0 TYPE ROOT_DIR [CONFIG_FILE]"
  echo "  TYPE: msa | boltz | esm"
  echo "  ROOT_DIR: Output directory to check (e.g. MSA output dir or output parent dir)"
  echo "  CONFIG_FILE: Optional; default: project root config.yaml"
  exit 1
fi

case "$TYPE" in
  msa|boltz|esm) ;;
  *)
    echo "ERROR: TYPE must be msa, boltz, or esm"
    exit 1
    ;;
esac

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: ROOT_DIR does not exist: $ROOT_DIR"
  exit 1
fi

ROOT_DIR="$(realpath -m "$ROOT_DIR")"

# Default config to project root
if [[ -z "${CONFIG_FILE:-}" ]]; then
  CONFIG_FILE="$(dirname "$SCRIPT_DIR")/config.yaml"
fi
if [[ -f "$CONFIG_FILE" ]]; then
  CONFIG_FILE="$(realpath -m "$CONFIG_FILE")"
  export CONFIG_FILE

  # Export same vars as run.sh so retry jobs get correct env
  SLURM_LOG_DIR=$(get_config "slurm.log_dir")
  MMSEQ2_DB=$(get_config "msa.mmseq2_db")
  COLABFOLD_DB=$(get_config "msa.colabfold_db")
  COLABFOLD_BIN=$(get_config "msa.colabfold_bin")
  BOLTZ_CACHE=$(get_config "boltz.cache_dir")
  BOLTZ_COLABFOLD_DB=$(get_config "boltz.colabfold_db")
  BOLTZ_ENV_PATH=$(get_config "boltz.env_path")
  ESM_ENV_PATH=$(get_config "esm.env_path")
  ESM_WORK_DIR=$(get_config "esm.work_dir")
  ESM_CACHE_DIR=$(get_config "esm.cache_dir")
  ES_SCRIPT_DIR=$(get_config "es.script_dir")
  ES_WT_PATH=$(get_config "es.wt_path")
  ES_OUTPUT_DIR=$(get_config "es.output_dir")
  ES_ENV_PATH=$(get_config "es.env_path")

  export SLURM_LOG_DIR MMSEQ2_DB COLABFOLD_DB
  export BOLTZ_CACHE BOLTZ_COLABFOLD_DB BOLTZ_ENV_PATH
  export ESM_ENV_PREFIX="$ESM_ENV_PATH" ESM_WORK_DIR ESM_CACHE_DIR
  export ES_SCRIPT_DIR ES_WT_PATH ES_OUTPUT_DIR ES_ENV_PREFIX="$ES_ENV_PATH"
  [[ -n "${COLABFOLD_BIN:-}" ]] && export PATH="${COLABFOLD_BIN}:${PATH}"
fi

export SCRIPT_DIR

case "$TYPE" in
  msa)  exec "${SCRIPT_DIR}/checker_msa.sh" "$ROOT_DIR" ;;
  boltz) exec "${SCRIPT_DIR}/checker_boltz.sh" "$ROOT_DIR" ;;
  esm)  exec "${SCRIPT_DIR}/checker_esm.sh" "$ROOT_DIR" ;;
esac
