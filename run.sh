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
SLURM_PARTITION=$(get_config "slurm.partition")
SLURM_ACCOUNT=$(get_config "slurm.account")
# Per-job partition (optional; default to SLURM_PARTITION)
SLURM_MSA_PARTITION=$(get_config "slurm.msa.partition"); [[ -z "$SLURM_MSA_PARTITION" ]] && SLURM_MSA_PARTITION="$SLURM_PARTITION"
SLURM_BOLTZ_PARTITION=$(get_config "slurm.boltz.partition"); [[ -z "$SLURM_BOLTZ_PARTITION" ]] && SLURM_BOLTZ_PARTITION="$SLURM_PARTITION"
SLURM_ESM_PARTITION=$(get_config "slurm.esm.partition"); [[ -z "$SLURM_ESM_PARTITION" ]] && SLURM_ESM_PARTITION="$SLURM_PARTITION"
SLURM_ES_PARTITION=$(get_config "slurm.es.partition"); [[ -z "$SLURM_ES_PARTITION" ]] && SLURM_ES_PARTITION="$SLURM_PARTITION"
SLURM_CHECKER_MSA_PARTITION=$(get_config "slurm.checker_msa.partition"); [[ -z "$SLURM_CHECKER_MSA_PARTITION" ]] && SLURM_CHECKER_MSA_PARTITION="$SLURM_MSA_PARTITION"
SLURM_CHECKER_BOLTZ_PARTITION=$(get_config "slurm.checker_boltz.partition"); [[ -z "$SLURM_CHECKER_BOLTZ_PARTITION" ]] && SLURM_CHECKER_BOLTZ_PARTITION="$SLURM_BOLTZ_PARTITION"
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
export SLURM_PARTITION SLURM_ACCOUNT
export SLURM_MSA_PARTITION SLURM_BOLTZ_PARTITION SLURM_ESM_PARTITION SLURM_ES_PARTITION
export SLURM_CHECKER_MSA_PARTITION SLURM_CHECKER_BOLTZ_PARTITION
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

MSA_JOB_ID=""
OUTPUT_DIR=""
if [[ "$RUN_MSA" -eq 1 ]]; then
  echo "Step 1: Launching MSA generation..."
  msa_output=$("$MSA_SCRIPT" "$INPUT_DIR" "$MSA_MAX_FILES_PER_JOB" "$OUTPUT_PARENT_DIR" "${MSA_ARRAY_MAX_CONCURRENCY:-100}")
  echo "$msa_output"
  MSA_ARRAY_JOB_ID=$(echo "$msa_output" | grep -E '^MSA_ARRAY_JOB_ID=' | sed 's/^MSA_ARRAY_JOB_ID=//')
  MSA_OUTPUT_DIR=$(echo "$msa_output" | grep -E '^MSA_OUTPUT_DIR=' | sed 's/^MSA_OUTPUT_DIR=//')
  MSA_JOB_ID="${MSA_ARRAY_JOB_ID}"
  OUTPUT_DIR="${MSA_OUTPUT_DIR}"
  if [[ -z "$MSA_JOB_ID" || -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: Failed to extract MSA job ID or output directory"
    echo "MSA output: $msa_output"
    exit 1
  fi
  echo "  MSA job ID: ${MSA_JOB_ID}"
  echo "  Output directory: ${OUTPUT_DIR}"
  echo "  Note: Post-processing (FASTA->YAML) will run automatically at the end of MSA job"
  echo ""

  if [[ -f "${SCRIPT_DIR}/run_checker_msa.slrm" ]]; then
    CHECKER_MSA_JOB_ID=$(sbatch --parsable --dependency=afternotok:"$MSA_JOB_ID" \
      ${SLURM_CHECKER_MSA_PARTITION:+-p "$SLURM_CHECKER_MSA_PARTITION"} ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} \
      -o "${SLURM_LOG_DIR}/%x.%j.out" \
      --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
      "${SCRIPT_DIR}/run_checker_msa.slrm" 2>/dev/null || echo "")
    [[ -n "$CHECKER_MSA_JOB_ID" ]] && echo "  MSA checker job ID: ${CHECKER_MSA_JOB_ID} (will run if MSA job fails)"
  fi
  echo ""

  echo "Step 2: Launching ESM embeddings and Boltz prediction (both depend on MSA job ${MSA_JOB_ID})..."

  if [[ "$RUN_ESM" -eq 1 ]]; then
    ESM_WRAPPER="${SCRIPT_DIR}/run_esm_wrapper.slrm"
    if [[ ! -f "$ESM_WRAPPER" ]]; then
      echo "ERROR: ESM wrapper script not found at $ESM_WRAPPER"
      exit 1
    fi
    ESM_JOB_ID=$(sbatch --parsable --dependency=afterok:"$MSA_JOB_ID" \
      ${SLURM_ESM_PARTITION:+-p "$SLURM_ESM_PARTITION"} ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} \
      -o "${SLURM_LOG_DIR}/%x.%j.out" \
      --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",OUTPUT_PARENT_DIR="$OUTPUT_PARENT_DIR",N="$ESM_N",ARRAY_MAX_CONCURRENCY="$ESM_ARRAY_MAX_CONCURRENCY",SCRIPT_DIR="$SCRIPT_DIR" \
      "$ESM_WRAPPER")
    if [[ -z "$ESM_JOB_ID" ]]; then
      echo "ERROR: Failed to submit ESM job"
      exit 1
    fi
    echo "  ESM job ID: ${ESM_JOB_ID}"
    CHECKER_ESM_SCRIPT="${SCRIPT_DIR}/checker_esm.sh"
    if [[ -f "$CHECKER_ESM_SCRIPT" ]]; then
      CHECKER_ESM_WRAPPER="${SCRIPT_DIR}/run_checker_esm.slrm"
      if [[ -f "$CHECKER_ESM_WRAPPER" ]]; then
        CHECKER_ESM_JOB_ID=$(sbatch --parsable --dependency=afternotok:"$ESM_JOB_ID" \
          ${SLURM_ESM_PARTITION:+-p "$SLURM_ESM_PARTITION"} ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} \
          -o "${SLURM_LOG_DIR}/%x.%j.out" \
          --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
          "$CHECKER_ESM_WRAPPER" 2>/dev/null || echo "")
        [[ -n "$CHECKER_ESM_JOB_ID" ]] && echo "  ESM checker job ID: ${CHECKER_ESM_JOB_ID} (will run if ESM job fails)"
      else
        CHECKER_ESM_JOB_ID=$(sbatch --parsable --dependency=afternotok:"$ESM_JOB_ID" \
          ${SLURM_ESM_PARTITION:+-p "$SLURM_ESM_PARTITION"} ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} \
          -o "${SLURM_LOG_DIR}/%x.%j.out" \
          --export=ALL,OUTPUT_DIR="$OUTPUT_DIR" \
          --wrap="bash $CHECKER_ESM_SCRIPT $OUTPUT_DIR" 2>/dev/null || echo "")
        [[ -n "$CHECKER_ESM_JOB_ID" ]] && echo "  ESM checker job ID: ${CHECKER_ESM_JOB_ID} (will run if ESM job fails)"
      fi
    fi
  fi

  if [[ "$RUN_BOLTZ" -eq 1 ]]; then
    BOLTZ_WRAPPER="${SCRIPT_DIR}/run_boltz_wrapper.slrm"
    if [[ ! -f "$BOLTZ_WRAPPER" ]]; then
      echo "ERROR: Boltz wrapper script not found at $BOLTZ_WRAPPER"
      exit 1
    fi
    BOLTZ_JOB_ID=$(sbatch --parsable --dependency=afterok:"$MSA_JOB_ID" \
      ${SLURM_BOLTZ_PARTITION:+-p "$SLURM_BOLTZ_PARTITION"} ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} \
      -o "${SLURM_LOG_DIR}/%x.%j.out" \
      --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",MAX_FILES_PER_JOB="$MAX_FILES_PER_JOB",ARRAY_MAX_CONCURRENCY="$BOLTZ_ARRAY_MAX_CONCURRENCY",SCRIPT_DIR="$SCRIPT_DIR",BOLTZ_RECYCLING_STEPS="$BOLTZ_RECYCLING_STEPS",BOLTZ_DIFFUSION_SAMPLES="$BOLTZ_DIFFUSION_SAMPLES" \
      "$BOLTZ_WRAPPER")
    if [[ -z "$BOLTZ_JOB_ID" ]]; then
      echo "ERROR: Failed to submit boltz job"
      exit 1
    fi
    echo "  Boltz job ID: ${BOLTZ_JOB_ID}"
    if [[ -f "${SCRIPT_DIR}/run_checker_boltz.slrm" ]]; then
      CHECKER_BOLTZ_JOB_ID=$(sbatch --parsable --dependency=afternotok:"$BOLTZ_JOB_ID" \
        ${SLURM_CHECKER_BOLTZ_PARTITION:+-p "$SLURM_CHECKER_BOLTZ_PARTITION"} ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} \
        -o "${SLURM_LOG_DIR}/%x.%j.out" \
        --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
        "${SCRIPT_DIR}/run_checker_boltz.slrm" 2>/dev/null || echo "")
      [[ -n "$CHECKER_BOLTZ_JOB_ID" ]] && echo "  Boltz checker job ID: ${CHECKER_BOLTZ_JOB_ID} (will run if Boltz job fails)"
    fi
  fi

  [[ "$RUN_ES" -eq 1 ]] && {
    echo ""
    echo ">>> Submitting ES analysis..."
    "${SCRIPT_DIR}/run_es.sh" "$OUTPUT_DIR"
  }
else
  OUTPUT_DIR="$INPUT_YAML_DIR"
  [[ "$RUN_BOLTZ" -eq 1 ]] && {
    echo ""
    echo ">>> Submitting Boltz array..."
    "$BOLTZ_SCRIPT" "$OUTPUT_DIR" "$MAX_FILES_PER_JOB" "$BOLTZ_ARRAY_MAX_CONCURRENCY" "$BOLTZ_RECYCLING_STEPS" "$BOLTZ_DIFFUSION_SAMPLES"
  }
  [[ "$RUN_ESM" -eq 1 ]] && {
    echo ""
    echo ">>> Submitting ESM array..."
    "$ESM_SCRIPT" "$OUTPUT_DIR" "$ESM_N" "$OUTPUT_PARENT_DIR" "$ESM_ARRAY_MAX_CONCURRENCY"
  }
  [[ "$RUN_ES" -eq 1 ]] && {
    echo ""
    echo ">>> Submitting ES analysis..."
    "${SCRIPT_DIR}/run_es.sh" "$OUTPUT_DIR"
  }
fi

echo ""
echo "==============================================="
echo "Orchestration submitted. Check SLURM for job status."
echo "To check for errors and retry: ./slurm_scripts/checker.sh msa|boltz|esm <output_dir> [config.yaml]"
echo "==============================================="
