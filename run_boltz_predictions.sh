#!/bin/bash
set -euo pipefail

# Run N independent Boltz prediction runs per sequence
# Reuses existing YAMLs (from MSA stage) — 1 MSA, N × Boltz runs
#
# Usage:
#   ./run_boltz_predictions.sh YAML_DIR NUM_RUNS [CONFIG_FILE]
#
# Examples:
#   ./run_boltz_predictions.sh /n/home06/tbush/outputs/sequences 5
#   ./run_boltz_predictions.sh /n/home06/tbush/outputs/sequences 10 config.yaml
#
# Output structure:
#   sequences/{seq_name}/boltz/
#     ├── run_0/
#     │   ├── {seq_name}_model_0.cif ... {seq_name}_model_24.cif
#     ├── run_1/
#     │   ├── {seq_name}_model_0.cif ... {seq_name}_model_24.cif
#     └── ...

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_DIR="${1:-}"
NUM_RUNS="${2:-}"
CONFIG_FILE="${3:-${ROOT_DIR}/config.yaml}"

if [[ -z "$YAML_DIR" || -z "$NUM_RUNS" ]]; then
  echo "Usage: $0 YAML_DIR NUM_RUNS [CONFIG_FILE]"
  echo ""
  echo "  YAML_DIR:    Directory containing .yaml files (e.g., outputs/sequences)"
  echo "  NUM_RUNS:    Number of independent Boltz runs per sequence"
  echo "  CONFIG_FILE: Path to config.yaml (default: ./config.yaml)"
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

if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]] || [[ "$NUM_RUNS" -lt 1 ]]; then
  echo "ERROR: NUM_RUNS must be a positive integer, got: $NUM_RUNS"
  exit 1
fi

PARSE_SCRIPT="${ROOT_DIR}/slurm_scripts/parse_config.py"

get_config() {
  python3 "$PARSE_SCRIPT" "$CONFIG_FILE" "$1" 2>/dev/null || echo ""
}

# Read config
MAX_FILES_PER_JOB=$(get_config "boltz.max_files_per_job")
MAX_FILES_PER_JOB="${MAX_FILES_PER_JOB:-25}"
RECYCLING_STEPS=$(get_config "boltz.recycling_steps")
RECYCLING_STEPS="${RECYCLING_STEPS:-10}"
DIFFUSION_SAMPLES=$(get_config "boltz.diffusion_samples")
DIFFUSION_SAMPLES="${DIFFUSION_SAMPLES:-25}"
CACHE_DIR=$(get_config "boltz.cache_dir")
ENV_PATH=$(get_config "boltz.env_path")
OUTPUT_DIR=$(get_config "output.parent_dir")

# SLURM settings
LOG_DIR=$(get_config "slurm.log_dir")
PARTITION=$(get_config "slurm.boltz.partition")
[[ -z "$PARTITION" ]] && PARTITION=$(get_config "slurm.partition")
ACCOUNT=$(get_config "slurm.account")
EMAIL=$(get_config "slurm.email")

# Container support
CONTAINER_SIF=$(get_config "containers.boltz")
BIND_PATHS=$(get_config "containers.bind_paths")
BIND_PATHS="${BIND_PATHS:-/n/holylfs06,/n/home06}"

# Validate
MISSING=()
[[ -z "$CACHE_DIR" ]] && MISSING+=("boltz.cache_dir")
[[ -z "$ENV_PATH" && -z "$CONTAINER_SIF" ]] && MISSING+=("boltz.env_path or containers.boltz")
[[ -z "$OUTPUT_DIR" ]] && MISSING+=("output.parent_dir")
[[ -z "$LOG_DIR" ]] && MISSING+=("slurm.log_dir")

if ((${#MISSING[@]} > 0)); then
  echo "ERROR: Missing required config values:"
  printf "  - %s\n" "${MISSING[@]}"
  exit 1
fi

# Resolve paths
YAML_DIR="$(realpath -m "$YAML_DIR")"
SEQUENCES_DIR="${OUTPUT_DIR}/sequences"
WORK_DIR="${OUTPUT_DIR}/boltz_multi_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$WORK_DIR" "$LOG_DIR"

# Find YAML files
mapfile -t YAML_FILES < <(find "$YAML_DIR" -name "*.yaml" -type f | sort)
TOTAL=${#YAML_FILES[@]}

if [[ "$TOTAL" -eq 0 ]]; then
  echo "ERROR: No .yaml files found in $YAML_DIR"
  exit 1
fi

# Create chunks (same chunking for all runs, each run gets its own output dir)
NUM_CHUNKS=$(( (TOTAL + MAX_FILES_PER_JOB - 1) / MAX_FILES_PER_JOB ))

for ((i=0; i<NUM_CHUNKS; i++)); do
  CHUNK_DIR="${WORK_DIR}/chunk_${i}"
  mkdir -p "$CHUNK_DIR"

  START=$((i * MAX_FILES_PER_JOB))
  END=$((START + MAX_FILES_PER_JOB))
  [[ "$END" -gt "$TOTAL" ]] && END=$TOTAL

  for ((j=START; j<END; j++)); do
    SRC="${YAML_FILES[$j]}"
    ln -sf "$SRC" "${CHUNK_DIR}/$(basename "$SRC")"
  done
done

echo "==============================================="
echo "Boltz Multi-Run Prediction Pipeline"
echo "==============================================="
echo "YAML dir:           $YAML_DIR"
echo "Sequences:          $TOTAL"
echo "Independent runs:   $NUM_RUNS"
echo "Diffusion samples:  $DIFFUSION_SAMPLES (per run)"
echo "Chunks:             $NUM_CHUNKS (max $MAX_FILES_PER_JOB per chunk)"
echo "Work dir:           $WORK_DIR"
echo ""
echo "Output structure:"
echo "  ${SEQUENCES_DIR}/{seq}/boltz/"
echo "    run_0/  ...  run_$((NUM_RUNS - 1))/"
echo "      model_0.cif ... model_$((DIFFUSION_SAMPLES - 1)).cif"
echo "==============================================="
echo ""

# Container prefix
CONTAINER_PREFIX=""
if [[ -n "$CONTAINER_SIF" ]]; then
  BIND_ARGS=""
  IFS=',' read -ra BPATHS <<< "$BIND_PATHS"
  for bp in "${BPATHS[@]}"; do
    BIND_ARGS+=" -B ${bp}"
  done
  CONTAINER_PREFIX="singularity exec --nv ${BIND_ARGS} ${CONTAINER_SIF}"
fi

# Submit prediction jobs: one array job per run
ALL_PREDICT_JOB_IDS=()

for ((run=0; run<NUM_RUNS; run++)); do
  ARRAY_RANGE="0-$((NUM_CHUNKS - 1))"
  JOB_NAME="boltz_run_${run}"

  PREDICT_JOB_ID=$(sbatch --parsable \
    --job-name="$JOB_NAME" \
    --array="$ARRAY_RANGE" \
    ${PARTITION:+-p "$PARTITION"} \
    ${ACCOUNT:+--account "$ACCOUNT"} \
    --cpus-per-task=8 \
    --mem=16G \
    --time=01:00:00 \
    --gpus-per-node=1 \
    --output="${LOG_DIR}/${JOB_NAME}.%A_%a.out" \
    --wrap="$(cat <<SLURM_EOF
set -euo pipefail
CHUNK_ID=\${SLURM_ARRAY_TASK_ID}
CHUNK_DIR="${WORK_DIR}/chunk_\${CHUNK_ID}"
CHUNK_OUTPUT="${WORK_DIR}/chunk_\${CHUNK_ID}_run_${run}_output"

export CUDA_VISIBLE_DEVICES=0
export NUM_GPU_DEVICES=1
export TMPDIR="\${SLURM_TMPDIR:-/tmp}"
export TRITON_CACHE_DIR="\${TMPDIR}/triton_cache_\${USER}_\${SLURM_JOB_ID}"
mkdir -p "\$TRITON_CACHE_DIR" "\$CHUNK_OUTPUT"

if [ -n "${CONTAINER_PREFIX}" ]; then
    ${CONTAINER_PREFIX} \\
        --env TRITON_CACHE_DIR=\$TRITON_CACHE_DIR \\
        boltz predict \$CHUNK_DIR \\
            --cache ${CACHE_DIR} --out_dir \$CHUNK_OUTPUT \\
            --devices 1 --accelerator gpu \\
            --recycling_steps ${RECYCLING_STEPS} \\
            --diffusion_samples ${DIFFUSION_SAMPLES} --override
else
    module load python/3.12.8-fasrc01 gcc/14.2.0-fasrc01 cuda/12.9.1-fasrc01 cudnn/9.10.2.21_cuda12-fasrc01 || true
    mamba activate ${ENV_PATH}
    boltz predict \$CHUNK_DIR \\
        --cache ${CACHE_DIR} --out_dir \$CHUNK_OUTPUT \\
        --devices 1 --accelerator gpu \\
        --recycling_steps ${RECYCLING_STEPS} \\
        --diffusion_samples ${DIFFUSION_SAMPLES} --override
fi

touch "\${CHUNK_OUTPUT}/.done"
echo "Run ${run}, chunk \${CHUNK_ID} complete"
SLURM_EOF
)")

  ALL_PREDICT_JOB_IDS+=("$PREDICT_JOB_ID")
  echo "Run $run: submitted array job ${PREDICT_JOB_ID} (chunks 0-$((NUM_CHUNKS - 1)))"
done

# Build dependency string: organize runs after ALL prediction jobs finish
DEPENDENCY_STR=$(IFS=:; echo "afterok:${ALL_PREDICT_JOB_IDS[*]}")

# Submit organize job
ORGANIZE_JOB_ID=$(sbatch --parsable \
  --job-name="boltz_organize" \
  --dependency="$DEPENDENCY_STR" \
  ${PARTITION:+-p "$PARTITION"} \
  ${ACCOUNT:+--account "$ACCOUNT"} \
  ${EMAIL:+--mail-type=END,FAIL --mail-user="$EMAIL"} \
  --cpus-per-task=4 \
  --mem=8G \
  --time=00:30:00 \
  --output="${LOG_DIR}/boltz_organize.%j.out" \
  --wrap="$(cat <<SLURM_EOF
set -euo pipefail
for RUN_ID in \$(seq 0 $((NUM_RUNS - 1))); do
  for CHUNK_ID in \$(seq 0 $((NUM_CHUNKS - 1))); do
    CHUNK_OUTPUT="${WORK_DIR}/chunk_\${CHUNK_ID}_run_\${RUN_ID}_output"
    if [[ -f "\${CHUNK_OUTPUT}/.done" ]]; then
      python3 ${ROOT_DIR}/workflow/scripts/organize_boltz_outputs.py \\
        --boltz_output_dir "\${CHUNK_OUTPUT}" \\
        --chunk_id "\${CHUNK_ID}" \\
        --sequences_dir "${SEQUENCES_DIR}" \\
        --subdirectory "run_\${RUN_ID}"
      echo "Organized run \${RUN_ID}, chunk \${CHUNK_ID}"
    else
      echo "WARNING: run \${RUN_ID}, chunk \${CHUNK_ID} not done, skipping"
    fi
  done
done
echo ""
echo "All runs organized. Output: ${SEQUENCES_DIR}/{seq}/boltz/run_{0..$((NUM_RUNS - 1))}/"
SLURM_EOF
)")

echo ""
echo "Submitted organize job: ${ORGANIZE_JOB_ID} (depends on all prediction jobs)"
echo ""
echo "==============================================="
echo "Summary"
echo "==============================================="
echo "Prediction jobs: ${ALL_PREDICT_JOB_IDS[*]}"
echo "Organize job:    ${ORGANIZE_JOB_ID}"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     ${LOG_DIR}/"
echo "Output:   ${SEQUENCES_DIR}/{seq}/boltz/run_{0..$((NUM_RUNS - 1))}/"
echo "==============================================="
