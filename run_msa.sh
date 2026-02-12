#!/bin/bash
set -euo pipefail

# Standalone MSA Pipeline
# Usage: ./run_msa.sh [CONFIG_FILE]
# Runs only the MSA generation stage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${ROOT_DIR}/slurm_scripts"
CONFIG_FILE="${1:-${ROOT_DIR}/config.yaml}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: Config file not found: $CONFIG_FILE"
  echo ""
  echo "Usage: $0 [CONFIG_FILE]"
  echo "  CONFIG_FILE: Path to YAML configuration file (default: config.yaml in project root)"
  exit 1
fi

PARSE_SCRIPT="${SCRIPT_DIR}/parse_config.py"

get_config() {
  local key="$1"
  python3 "$PARSE_SCRIPT" "$CONFIG_FILE" "$key" 2>/dev/null || echo ""
}

# Read configuration values
INPUT_DIR=$(get_config "input.fasta_dir")
OUTPUT_PARENT_DIR=$(get_config "output.parent_dir")
MSA_MAX_FILES_PER_JOB=$(get_config "msa.max_files_per_job")
MSA_ARRAY_MAX_CONCURRENCY=$(get_config "msa.array_max_concurrency")

# SLURM settings
SLURM_LOG_DIR=$(get_config "slurm.log_dir")
SLURM_PARTITION=$(get_config "slurm.partition")
SLURM_ACCOUNT=$(get_config "slurm.account")
SLURM_MSA_PARTITION=$(get_config "slurm.msa.partition")
[[ -z "$SLURM_MSA_PARTITION" ]] && SLURM_MSA_PARTITION="$SLURM_PARTITION"
SLURM_CHECKER_MSA_PARTITION=$(get_config "slurm.checker_msa.partition")
[[ -z "$SLURM_CHECKER_MSA_PARTITION" ]] && SLURM_CHECKER_MSA_PARTITION="$SLURM_MSA_PARTITION"

# MSA-specific paths
MMSEQ2_DB=$(get_config "msa.mmseq2_db")
COLABFOLD_DB=$(get_config "msa.colabfold_db")
COLABFOLD_BIN=$(get_config "msa.colabfold_bin")

# Set defaults
MSA_ARRAY_MAX_CONCURRENCY="${MSA_ARRAY_MAX_CONCURRENCY:-100}"

# Validate required parameters
MISSING_PARAMS=()
[[ -z "$INPUT_DIR" ]] && MISSING_PARAMS+=("input.fasta_dir")
[[ -z "$OUTPUT_PARENT_DIR" ]] && MISSING_PARAMS+=("output.parent_dir")
[[ -z "$MSA_MAX_FILES_PER_JOB" ]] && MISSING_PARAMS+=("msa.max_files_per_job")
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

MSA_SCRIPT="${SCRIPT_DIR}/split_and_run_msa.sh"

# Normalize paths
OUTPUT_PARENT_DIR="$(realpath -m "$OUTPUT_PARENT_DIR")"
CONFIG_FILE="$(realpath -m "$CONFIG_FILE")"
INPUT_DIR="$(realpath -m "$INPUT_DIR")"

# Export config-derived env for child scripts
export CONFIG_FILE SLURM_LOG_DIR
export SLURM_PARTITION SLURM_ACCOUNT SLURM_MSA_PARTITION SLURM_CHECKER_MSA_PARTITION
export MMSEQ2_DB COLABFOLD_DB
[[ -n "$COLABFOLD_BIN" ]] && export PATH="${COLABFOLD_BIN}:${PATH}"

echo "==============================================="
echo "Standalone MSA Pipeline"
echo "==============================================="
echo "Input dir: $INPUT_DIR"
echo "Output parent dir: $OUTPUT_PARENT_DIR"
echo "Config: $CONFIG_FILE"
echo ""

echo "Launching MSA generation..."
msa_output=$("$MSA_SCRIPT" "$INPUT_DIR" "$MSA_MAX_FILES_PER_JOB" "$OUTPUT_PARENT_DIR" "${MSA_ARRAY_MAX_CONCURRENCY}")
echo "$msa_output"

MSA_ARRAY_JOB_ID=$(echo "$msa_output" | grep -E '^MSA_ARRAY_JOB_ID=' | sed 's/^MSA_ARRAY_JOB_ID=//')
MSA_OUTPUT_DIR=$(echo "$msa_output" | grep -E '^MSA_OUTPUT_DIR=' | sed 's/^MSA_OUTPUT_DIR=//')
MSA_JOB_ID="${MSA_ARRAY_JOB_ID}"
OUTPUT_DIR="${MSA_OUTPUT_DIR}"

if [[ -z "$MSA_JOB_ID" || -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: Failed to extract MSA job ID or output directory"
  exit 1
fi

echo ""
echo "MSA job ID: ${MSA_JOB_ID}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Submit checker job
if [[ -f "${SCRIPT_DIR}/run_checker_msa.slrm" ]]; then
  CHECKER_MSA_JOB_ID=$(sbatch --parsable --dependency=afternotok:"$MSA_JOB_ID" \
    ${SLURM_CHECKER_MSA_PARTITION:+-p "$SLURM_CHECKER_MSA_PARTITION"} ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} \
    -o "${SLURM_LOG_DIR}/%x.%j.out" \
    --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
    "${SCRIPT_DIR}/run_checker_msa.slrm" 2>/dev/null || echo "")
  [[ -n "$CHECKER_MSA_JOB_ID" ]] && echo "MSA checker job ID: ${CHECKER_MSA_JOB_ID} (will run if MSA job fails)"
fi

echo ""
echo "==============================================="
echo "MSA pipeline submitted."
echo "Output directory: $OUTPUT_DIR"
echo "To check for errors and retry: ./slurm_scripts/checker.sh msa $OUTPUT_DIR $CONFIG_FILE"
echo "==============================================="
