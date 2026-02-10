#!/bin/bash
set -euo pipefail

# Usage: ./run_msa_then_boltz.sh [CONFIG_FILE]
# Example: ./run_msa_then_boltz.sh pipeline_config.yaml
#          ./run_msa_then_boltz.sh  (uses pipeline_config.yaml in script directory)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

# Parse config file using Python
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

# Step 1: Launch MSA array job
echo "Step 1: Launching MSA generation..."
MSA_JOB_OUTPUT=$("$MSA_SCRIPT" "$INPUT_DIR" "$MSA_MAX_FILES_PER_JOB" "$OUTPUT_PARENT_DIR" "$MSA_ARRAY_MAX_CONCURRENCY")

# Extract MSA job ID and output directory from the output
MSA_JOB_ID=$(echo "$MSA_JOB_OUTPUT" | grep -oP 'Submitted array job \K[0-9]+' || echo "")
OUTPUT_DIR=$(echo "$MSA_JOB_OUTPUT" | grep -oP 'Chunks dir: \K[^ ]+' || echo "")

if [[ -z "$MSA_JOB_ID" || -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: Failed to extract MSA job ID or output directory"
  echo "MSA output: $MSA_JOB_OUTPUT"
  exit 1
fi

echo "  MSA job ID: ${MSA_JOB_ID}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Note: Post-processing (FASTA->YAML) will run automatically at the end of MSA job"
echo ""

# Submit MSA checker job (runs only if MSA job fails)
CHECKER_MSA_SCRIPT="${SCRIPT_DIR}/run_checker_msa.slrm"
if [[ -f "$CHECKER_MSA_SCRIPT" ]]; then
  CHECKER_MSA_JOB_ID=$(sbatch --parsable \
    --dependency=afternotok:${MSA_JOB_ID} \
    --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
    "$CHECKER_MSA_SCRIPT" 2>/dev/null || echo "")
  if [[ -n "$CHECKER_MSA_JOB_ID" ]]; then
    echo "  MSA checker job ID: ${CHECKER_MSA_JOB_ID} (will run if MSA job fails)"
  fi
fi
echo ""

# Step 2: Launch ESM and Boltz array jobs that depend on MSA completion
# Post-processing is now integrated into the MSA job, so both depend directly on MSA
echo "Step 2: Launching ESM embeddings and Boltz prediction (both depend on MSA job ${MSA_JOB_ID})..."

# Launch ESM job
ESM_WRAPPER="${SCRIPT_DIR}/run_esm_wrapper.slrm"
if [[ ! -f "$ESM_WRAPPER" ]]; then
  echo "ERROR: ESM wrapper script not found at $ESM_WRAPPER"
  exit 1
fi

ESM_WRAPPER_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:${MSA_JOB_ID} \
  --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",N="$ESM_N",ARRAY_MAX_CONCURRENCY="$ESM_ARRAY_MAX_CONCURRENCY",SCRIPT_DIR="$SCRIPT_DIR" \
  "$ESM_WRAPPER")

ESM_JOB_ID="$ESM_WRAPPER_JOB_ID"

if [[ -z "$ESM_JOB_ID" ]]; then
  echo "ERROR: Failed to submit ESM job"
  exit 1
fi

echo "  ESM job ID: ${ESM_JOB_ID}"

# Submit ESM checker job (runs only if ESM job fails)
CHECKER_ESM_SCRIPT="${SCRIPT_DIR}/checker_esm.sh"
if [[ -f "$CHECKER_ESM_SCRIPT" ]]; then
  CHECKER_ESM_WRAPPER="${SCRIPT_DIR}/run_checker_esm.slrm"
  if [[ -f "$CHECKER_ESM_WRAPPER" ]]; then
    CHECKER_ESM_JOB_ID=$(sbatch --parsable \
      --dependency=afternotok:${ESM_JOB_ID} \
      --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
      "$CHECKER_ESM_WRAPPER" 2>/dev/null || echo "")
    if [[ -n "$CHECKER_ESM_JOB_ID" ]]; then
      echo "  ESM checker job ID: ${CHECKER_ESM_JOB_ID} (will run if ESM job fails)"
    fi
  else
    # Fallback: submit checker directly if no wrapper exists
    CHECKER_ESM_JOB_ID=$(sbatch --parsable \
      --dependency=afternotok:${ESM_JOB_ID} \
      --export=ALL,OUTPUT_DIR="$OUTPUT_DIR" \
      --wrap="bash $CHECKER_ESM_SCRIPT $OUTPUT_DIR" 2>/dev/null || echo "")
    if [[ -n "$CHECKER_ESM_JOB_ID" ]]; then
      echo "  ESM checker job ID: ${CHECKER_ESM_JOB_ID} (will run if ESM job fails)"
    fi
  fi
fi

# Launch Boltz job
BOLTZ_WRAPPER="${SCRIPT_DIR}/run_boltz_wrapper.slrm"
if [[ ! -f "$BOLTZ_WRAPPER" ]]; then
  echo "ERROR: Boltz wrapper script not found at $BOLTZ_WRAPPER"
  exit 1
fi

# Submit boltz wrapper job with dependency on MSA completion
# Post-processing is integrated into MSA job, so we depend on MSA job directly
# Export SCRIPT_DIR so wrapper can find split_and_run_boltz.sh
BOLTZ_WRAPPER_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:${MSA_JOB_ID} \
  --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",MAX_FILES_PER_JOB="$MAX_FILES_PER_JOB",ARRAY_MAX_CONCURRENCY="$BOLTZ_ARRAY_MAX_CONCURRENCY",SCRIPT_DIR="$SCRIPT_DIR",BOLTZ_RECYCLING_STEPS="$BOLTZ_RECYCLING_STEPS",BOLTZ_DIFFUSION_SAMPLES="$BOLTZ_DIFFUSION_SAMPLES" \
  "$BOLTZ_WRAPPER")

BOLTZ_JOB_ID="$BOLTZ_WRAPPER_JOB_ID"

if [[ -z "$BOLTZ_JOB_ID" ]]; then
  echo "ERROR: Failed to submit boltz job"
  exit 1
fi

echo "  Boltz job ID: ${BOLTZ_JOB_ID}"

# Submit Boltz+ES checker job (runs only if Boltz job fails)
CHECKER_BOLTZ_SCRIPT="${SCRIPT_DIR}/run_checker_boltz.slrm"
if [[ -f "$CHECKER_BOLTZ_SCRIPT" ]]; then
  CHECKER_BOLTZ_JOB_ID=$(sbatch --parsable \
    --dependency=afternotok:${BOLTZ_JOB_ID} \
    --export=ALL,OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
    "$CHECKER_BOLTZ_SCRIPT" 2>/dev/null || echo "")
  if [[ -n "$CHECKER_BOLTZ_JOB_ID" ]]; then
    echo "  Boltz+ES checker job ID: ${CHECKER_BOLTZ_JOB_ID} (will run if Boltz job fails)"
  fi
fi
echo ""

ES_SCRIPT="$SCRIPT_DIR/run_es_parallel.slrm"

ES_JOB_ID=$(sbatch "$ES_SCRIPT" "$OUTPUT_DIR" "$ES_SCRIPT_DIR" "$ES_WT_PATH")

echo "==============================================="
echo "All jobs submitted successfully:"
echo "  MSA job: ${MSA_JOB_ID} (includes post-processing: FASTA->YAML)"
if [[ -n "${CHECKER_MSA_JOB_ID:-}" ]]; then
  echo "    Checker: ${CHECKER_MSA_JOB_ID} (runs if MSA job fails)"
fi
echo "  ESM job: ${ESM_JOB_ID} (depends on MSA job, runs in parallel with Boltz)"
if [[ -n "${CHECKER_ESM_JOB_ID:-}" ]]; then
  echo "    Checker: ${CHECKER_ESM_JOB_ID} (runs if ESM job fails)"
fi
echo "  Boltz job: ${BOLTZ_JOB_ID} (depends on MSA job, runs in parallel with ESM)"
if [[ -n "${CHECKER_BOLTZ_JOB_ID:-}" ]]; then
  echo "    Checker: ${CHECKER_BOLTZ_JOB_ID} (runs if Boltz job fails, checks both Boltz and ES)"
fi
echo "==============================================="
echo ""
echo "Monitor jobs with:"
if [[ -n "$ES_JOB_ID" ]]; then
  echo "  squeue -u $USER"
  echo "  squeue -j ${MSA_JOB_ID},${ESM_JOB_ID},${BOLTZ_JOB_ID},${ES_JOB_ID}"
else
  echo "  squeue -u $USER"
  echo "  squeue -j ${MSA_JOB_ID},${ESM_JOB_ID},${BOLTZ_JOB_ID}"
fi
