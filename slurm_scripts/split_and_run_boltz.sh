#!/bin/bash
set -euo pipefail

# Usage: ./split_and_run_boltz.sh OUTPUT_DIR MAX_FILES_PER_JOB [ARRAY_MAX_CONCURRENCY] [RECYCLING_STEPS] [DIFFUSION_SAMPLES]
# Example: ./split_and_run_boltz.sh /data/chunks_20240101_120000 10 100 8 10

OUTPUT_DIR="${1:-}"
MAX_FILES_PER_JOB="${2:-}"
ARRAY_MAX_CONCURRENCY="${3:-100}"
RECYCLING_STEPS="${4:-8}"
DIFFUSION_SAMPLES="${5:-10}"

if [[ -z "${OUTPUT_DIR}" || -z "${MAX_FILES_PER_JOB}" ]]; then
  echo "Usage: $0 OUTPUT_DIR MAX_FILES_PER_JOB [ARRAY_MAX_CONCURRENCY] [RECYCLING_STEPS] [DIFFUSION_SAMPLES]"
  echo "  OUTPUT_DIR: Directory containing sequence folders with .yaml files"
  echo "  MAX_FILES_PER_JOB: Maximum number of .yaml files per array job"
  echo "  ARRAY_MAX_CONCURRENCY: Maximum concurrent array tasks (default: 100)"
  echo "  RECYCLING_STEPS: Number of recycling steps for boltz prediction (default: 8)"
  echo "  DIFFUSION_SAMPLES: Number of diffusion samples for boltz prediction (default: 10)"
  exit 1
fi
if ! [[ "$MAX_FILES_PER_JOB" =~ ^[0-9]+$ && "$MAX_FILES_PER_JOB" -ge 1 ]]; then
  echo "MAX_FILES_PER_JOB must be a positive integer"
  exit 1
fi

# Normalize to absolute path
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

# Make timestamped output directory for chunks
TS="$(date +%Y%m%d_%H%M%S)"
CHUNKS_DIR="${OUTPUT_DIR}/boltz_chunks_${TS}"
mkdir -p "$CHUNKS_DIR"

echo "Finding .yaml files in: $OUTPUT_DIR"
echo "Chunk directories will be created in: $CHUNKS_DIR"

# Recursively collect all .yaml files (absolute paths), null-safe & sorted
mapfile -d '' -t yaml_files < <(find "$OUTPUT_DIR" -type f -name "*.yaml" -print0 | sort -z)
total=${#yaml_files[@]}

if ((total == 0)); then
  echo "No .yaml files found in $OUTPUT_DIR"
  exit 1
fi

echo "Found ${total} .yaml files"

# Calculate number of chunks needed (ceil division)
NUM_CHUNKS=$(((total + MAX_FILES_PER_JOB - 1) / MAX_FILES_PER_JOB))

echo "Creating ${NUM_CHUNKS} chunk directories (max ${MAX_FILES_PER_JOB} files per chunk)..."

# Create chunk directories and copy yaml files
for ((i = 0; i < NUM_CHUNKS; i++)); do
  start=$((i * MAX_FILES_PER_JOB))
  end=$((start + MAX_FILES_PER_JOB))
  ((end > total)) && end=$total
  ((start >= end)) && continue # skip empty chunk

  CHUNK_DIR="${CHUNKS_DIR}/chunk_${i}"
  mkdir -p "$CHUNK_DIR"

  # Copy yaml files to chunk directory
  for ((j = start; j < end; j++)); do
    cp "${yaml_files[j]}" "$CHUNK_DIR/"
  done

  file_count=$((end - start))
  echo "Created chunk_${i} with ${file_count} files"
done

# -------- Create tot_filesboltz.txt and processed_paths.txt --------
echo "Creating tot_filesboltz.txt and processed_paths.txt..."

TOT_FILES_BOLTZ="${CHUNKS_DIR}/tot_filesboltz.txt"
PROCESSED_PATHS_FILE="${CHUNKS_DIR}/processed_paths.txt"

# Write all yaml file paths to tot_filesboltz.txt
: >"$TOT_FILES_BOLTZ"
for yaml_file in "${yaml_files[@]}"; do
  echo "$yaml_file" >>"$TOT_FILES_BOLTZ"
done
sort -u "$TOT_FILES_BOLTZ" -o "$TOT_FILES_BOLTZ"

# Create empty processed_paths.txt
: >"$PROCESSED_PATHS_FILE"

echo "Created $TOT_FILES_BOLTZ with $(wc -l <"$TOT_FILES_BOLTZ") total paths"
echo "Created empty $PROCESSED_PATHS_FILE"

# -------- Build manifest (stable order) & submit array --------
echo "Building manifest and submitting array..."

MANIFEST="${CHUNKS_DIR}/chunkdirs.manifest"
: >"$MANIFEST"

# Deterministic: sort by name; include only non-empty chunk directories
while IFS= read -r -d '' chunk_dir; do
  if [[ -d "$chunk_dir" ]] && [[ -n "$(find "$chunk_dir" -maxdepth 1 -name "*.yaml" -type f)" ]]; then
    realpath -s "$chunk_dir" >>"$MANIFEST"
  fi
done < <(find "$CHUNKS_DIR" -maxdepth 1 -type d -name 'chunk_*' -print0 | sort -z)

NUM_TASKS=$(wc -l <"$MANIFEST")
if ((NUM_TASKS == 0)); then
  echo "No non-empty chunk directories found; nothing to submit."
  exit 0
fi

echo "Submitting ${NUM_TASKS} array tasks (max concurrent: ${ARRAY_MAX_CONCURRENCY})..."

# Get script directory to find run_boltz_array.slrm
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOLTZ_SCRIPT="${SCRIPT_DIR}/run_boltz_array.slrm"

if [[ ! -f "$BOLTZ_SCRIPT" ]]; then
  echo "ERROR: Boltz array script not found at $BOLTZ_SCRIPT"
  exit 1
fi

# --parsable returns just the job ID; pass cache/env from config (run.sh exports them)
SLURM_OUTPUT="${SLURM_LOG_DIR:-/tmp}/%x.%A_%a.out"
ARRAY_JOB_ID="$(
  sbatch --parsable \
    -o "$SLURM_OUTPUT" \
    --array=1-"$NUM_TASKS"%${ARRAY_MAX_CONCURRENCY} \
    --export=ALL,MANIFEST="$MANIFEST",BASE_OUTPUT_DIR="$CHUNKS_DIR",BOLTZ_RECYCLING_STEPS="$RECYCLING_STEPS",BOLTZ_DIFFUSION_SAMPLES="$DIFFUSION_SAMPLES",CONFIG_FILE="${CONFIG_FILE:-}",BOLTZ_CACHE="${BOLTZ_CACHE:-}",BOLTZ_COLABFOLD_DB="${BOLTZ_COLABFOLD_DB:-}",BOLTZ_ENV_PATH="${BOLTZ_ENV_PATH:-}",COLABFOLD_BIN="${COLABFOLD_BIN:-}" \
    "$BOLTZ_SCRIPT"
)"

echo "Submitted array job ${ARRAY_JOB_ID} with ${NUM_TASKS} tasks."
echo "Chunks dir: $CHUNKS_DIR"
echo "Manifest:   $MANIFEST"
echo "Total files: $(wc -l <"$TOT_FILES_BOLTZ")"

# Submit post-processing job to organize outputs after array job completes
ORGANIZE_SCRIPT="${SCRIPT_DIR}/run_boltz_organize.slrm"
if [[ -f "$ORGANIZE_SCRIPT" ]]; then
  echo ""
  echo "Submitting post-processing job to organize boltz outputs..."
  ORGANIZE_OUTPUT="${SLURM_LOG_DIR:-/tmp}/%x.%j.out"
  ORGANIZE_JOB_ID=$(sbatch --parsable \
    -o "$ORGANIZE_OUTPUT" \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --export=ALL,BASE_OUTPUT_DIR="$OUTPUT_DIR",BOLTZ_CHUNKS_DIR="$CHUNKS_DIR",SCRIPT_DIR="$SCRIPT_DIR",CONFIG_FILE="${CONFIG_FILE:-}" \
    "$ORGANIZE_SCRIPT")

  if [[ -n "$ORGANIZE_JOB_ID" ]]; then
    echo "Submitted organize job ${ORGANIZE_JOB_ID} (depends on array job ${ARRAY_JOB_ID})"
  else
    echo "WARNING: Failed to submit organize job"
  fi
else
  echo "WARNING: Organize script not found at $ORGANIZE_SCRIPT, skipping post-processing"
fi
