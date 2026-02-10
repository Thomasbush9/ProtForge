#!/bin/bash
set -euo pipefail

# Usage: ./run.sh [CONFIG_FILE]
# Example: ./run.sh
#          ./run.sh /path/to/config.yaml

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${ROOT_DIR}/slurm_scripts"
CONFIG_FILE="${1:-${ROOT_DIR}/config.yaml}"

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: Config file not found: $CONFIG_FILE"
  echo ""
  echo "Usage: $0 [CONFIG_FILE]"
  echo "  CONFIG_FILE: Path to YAML configuration file (default: config.yaml in project root)"
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
INPUT_YAML_DIR=$(get_config "input.yaml_dir")
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

# Pipeline stage toggles (missing or non-"false" => true)
_run_flag() { [[ "$(get_config "pipeline.$1")" == "false" ]] && echo "0" || echo "1"; }
RUN_MSA=$(_run_flag "msa")
RUN_BOLTZ=$(_run_flag "boltz")
RUN_ESM=$(_run_flag "esm")
RUN_ES=$(_run_flag "es")

# Per-feature cache/path keys from config
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
ES_OUTPUT_DIR=$(get_config "es.output_dir")
ES_ENV_PATH=$(get_config "es.env_path")

# Set defaults
ARRAY_MAX_CONCURRENCY="${MSA_ARRAY_MAX_CONCURRENCY:-100}"
BOLTZ_ARRAY_MAX_CONCURRENCY="${BOLTZ_ARRAY_MAX_CONCURRENCY:-${ARRAY_MAX_CONCURRENCY}}"
ESM_ARRAY_MAX_CONCURRENCY="${ESM_ARRAY_MAX_CONCURRENCY:-${ARRAY_MAX_CONCURRENCY}}"
ES_ARRAY_MAX_CONCURRENCY="${ES_ARRAY_MAX_CONCURRENCY:-10}"

# Pipeline rule validation
if [[ "$RUN_MSA" -eq 0 && -z "$INPUT_YAML_DIR" ]]; then
  echo "ERROR: pipeline.msa is false but input.yaml_dir is not set (required when skipping MSA)"
  exit 1
fi
if [[ "$RUN_ES" -eq 1 && "$RUN_BOLTZ" -eq 0 ]]; then
  echo "ERROR: pipeline.es is true but pipeline.boltz is false (ES requires .cif from Boltz)"
  exit 1
fi

# Validate required parameters for enabled stages only
MISSING_PARAMS=()
[[ -z "$OUTPUT_PARENT_DIR" ]] && MISSING_PARAMS+=("output.parent_dir")
[[ -z "$SLURM_LOG_DIR" ]] && MISSING_PARAMS+=("slurm.log_dir")
[[ "$RUN_MSA" -eq 1 ]] && {
  [[ -z "$INPUT_DIR" ]] && MISSING_PARAMS+=("input.fasta_dir")
  [[ -z "$MSA_MAX_FILES_PER_JOB" ]] && MISSING_PARAMS+=("msa.max_files_per_job")
}
[[ "$RUN_BOLTZ" -eq 1 ]] && [[ -z "$MAX_FILES_PER_JOB" ]] && MISSING_PARAMS+=("boltz.max_files_per_job")
[[ "$RUN_ESM" -eq 1 ]] && [[ -z "$ESM_N" ]] && MISSING_PARAMS+=("esm.num_chunks")

if ((${#MISSING_PARAMS[@]} > 0)); then
  echo "ERROR: Missing required parameters in config file:"
  for param in "${MISSING_PARAMS[@]}"; do
    echo "  - $param"
  done
  echo ""
  echo "Config file: $CONFIG_FILE"
  exit 1
fi

MSA_SCRIPT="${SCRIPT_DIR}/split_and_run_msa.sh"
BOLTZ_SCRIPT="${SCRIPT_DIR}/split_and_run_boltz.sh"
ESM_SCRIPT="${SCRIPT_DIR}/run_esm.sh"

# Normalize paths
OUTPUT_PARENT_DIR="$(realpath -m "$OUTPUT_PARENT_DIR")"
CONFIG_FILE="$(realpath -m "$CONFIG_FILE")"
[[ -n "$INPUT_DIR" ]] && INPUT_DIR="$(realpath -m "$INPUT_DIR")"
[[ -n "$INPUT_YAML_DIR" ]] && INPUT_YAML_DIR="$(realpath -m "$INPUT_YAML_DIR")"

# Export config-derived env for child scripts and sbatch
export CONFIG_FILE SLURM_LOG_DIR
export MMSEQ2_DB COLABFOLD_DB
export BOLTZ_CACHE BOLTZ_COLABFOLD_DB BOLTZ_ENV_PATH
export ESM_ENV_PREFIX="$ESM_ENV_PATH" ESM_WORK_DIR ESM_CACHE_DIR
export ES_OUTPUT_DIR ES_SCRIPT_DIR ES_WT_PATH ES_ENV_PREFIX="$ES_ENV_PATH"
[[ -n "$COLABFOLD_BIN" ]] && export PATH="${COLABFOLD_BIN}:${PATH}"

echo "==============================================="
echo "Orchestrating pipeline (msa=$RUN_MSA boltz=$RUN_BOLTZ esm=$RUN_ESM es=$RUN_ES)"
echo "==============================================="
[[ "$RUN_MSA" -eq 1 ]] && echo "Input dir: $INPUT_DIR"
[[ "$RUN_MSA" -eq 0 ]] && echo "YAML dir: $INPUT_YAML_DIR"
echo "Output parent dir: $OUTPUT_PARENT_DIR"
echo "Config: $CONFIG_FILE"
echo ""

MSA_ARRAY_JOB_ID=""
if [[ "$RUN_MSA" -eq 1 ]]; then
  echo ">>> Step 1: Submitting MSA array..."
  msa_output=$("$MSA_SCRIPT" "$INPUT_DIR" "$MSA_MAX_FILES_PER_JOB" "$OUTPUT_PARENT_DIR" "$ARRAY_MAX_CONCURRENCY")
  echo "$msa_output"
  MSA_ARRAY_JOB_ID=$(echo "$msa_output" | grep -E '^MSA_ARRAY_JOB_ID=' | sed 's/^MSA_ARRAY_JOB_ID=//')
  MSA_OUTPUT_DIR=$(echo "$msa_output" | grep -E '^MSA_OUTPUT_DIR=' | sed 's/^MSA_OUTPUT_DIR=//')
  if [[ -z "$MSA_ARRAY_JOB_ID" || -z "$MSA_OUTPUT_DIR" ]]; then
    echo "ERROR: Could not get MSA_ARRAY_JOB_ID or MSA_OUTPUT_DIR from MSA script output"
    exit 1
  fi
  # MSA checker (runs only if MSA array fails)
  if [[ -f "${SCRIPT_DIR}/run_checker_msa.slrm" ]]; then
    CHECKER_MSA_ID=$(sbatch --parsable --dependency=afternotok:"$MSA_ARRAY_JOB_ID" \
      -o "${SLURM_LOG_DIR}/%x.%j.out" \
      --export=ALL,OUTPUT_DIR="$MSA_OUTPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
      "${SCRIPT_DIR}/run_checker_msa.slrm" 2>/dev/null || echo "")
    [[ -n "$CHECKER_MSA_ID" ]] && echo "  MSA checker job: ${CHECKER_MSA_ID} (runs if MSA fails)"
  fi
  echo ">>> Submitting Boltz and ESM wrappers with --dependency=afterok:${MSA_ARRAY_JOB_ID}..."
  [[ "$RUN_BOLTZ" -eq 1 ]] && {
    BOLTZ_WRAPPER_ID=$(sbatch --parsable --dependency=afterok:"$MSA_ARRAY_JOB_ID" \
      -o "${SLURM_LOG_DIR}/%x.%j.out" \
      --export=ALL,MSA_OUTPUT_DIR="$MSA_OUTPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
      "${SCRIPT_DIR}/run_boltz_wrapper.slrm")
    echo "  Boltz wrapper job: ${BOLTZ_WRAPPER_ID}"
    if [[ -f "${SCRIPT_DIR}/run_checker_boltz.slrm" ]]; then
      CHECKER_BOLTZ_ID=$(sbatch --parsable --dependency=afternotok:"$BOLTZ_WRAPPER_ID" \
        -o "${SLURM_LOG_DIR}/%x.%j.out" \
        --export=ALL,OUTPUT_DIR="$MSA_OUTPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
        "${SCRIPT_DIR}/run_checker_boltz.slrm" 2>/dev/null || echo "")
      [[ -n "$CHECKER_BOLTZ_ID" ]] && echo "  Boltz checker job: ${CHECKER_BOLTZ_ID} (runs if Boltz wrapper fails)"
    fi
  }
  [[ "$RUN_ESM" -eq 1 ]] && {
    ESM_WRAPPER_ID=$(sbatch --parsable --dependency=afterok:"$MSA_ARRAY_JOB_ID" \
      -o "${SLURM_LOG_DIR}/%x.%j.out" \
      --export=ALL,MSA_OUTPUT_DIR="$MSA_OUTPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
      "${SCRIPT_DIR}/run_esm_wrapper.slrm")
    echo "  ESM wrapper job: ${ESM_WRAPPER_ID}"
  }
  [[ "$RUN_ES" -eq 1 ]] && {
    echo ""
    echo ">>> Submitting ES analysis..."
    "${SCRIPT_DIR}/run_es.sh" "$MSA_OUTPUT_DIR"
  }
else
  MSA_OUTPUT_DIR="$INPUT_YAML_DIR"
  [[ "$RUN_BOLTZ" -eq 1 ]] && {
    echo ""
    echo ">>> Submitting Boltz array..."
    "$BOLTZ_SCRIPT" "$MSA_OUTPUT_DIR" "$MAX_FILES_PER_JOB" "$BOLTZ_ARRAY_MAX_CONCURRENCY" "$BOLTZ_RECYCLING_STEPS" "$BOLTZ_DIFFUSION_SAMPLES"
  }
  [[ "$RUN_ESM" -eq 1 ]] && {
    echo ""
    echo ">>> Submitting ESM array..."
    "$ESM_SCRIPT" "$MSA_OUTPUT_DIR" "$ESM_N" "$OUTPUT_PARENT_DIR" "$ESM_ARRAY_MAX_CONCURRENCY"
  }
  [[ "$RUN_ES" -eq 1 ]] && {
    echo ""
    echo ">>> Submitting ES analysis..."
    "${SCRIPT_DIR}/run_es.sh" "$MSA_OUTPUT_DIR"
  }
fi

echo ""
echo "==============================================="
echo "Orchestration submitted. Check SLURM for job status."
echo "To check for errors and retry: ./slurm_scripts/checker.sh msa|boltz|esm <output_dir> [config.yaml]"
echo "==============================================="
