#!/bin/bash
set -euo pipefail

# Usage: ./split_and_run_msa.sh INPUT_DIR MAX_FILES_PER_JOB OUTPUT_PARENT_DIR [ARRAY_MAX_CONCURRENCY]
# Example: ./split_and_run_msa.sh /data/fasta 25 /data/jobs 100

INPUT_DIR="${1:-}"
MAX_FILES_PER_JOB="${2:-}"
OUTPUT_PARENT_DIR="${3:-}"
ARRAY_MAX_CONCURRENCY="${4:-100}"

if [[ -z "${INPUT_DIR}" || -z "${MAX_FILES_PER_JOB}" || -z "${OUTPUT_PARENT_DIR}" ]]; then
  echo "Usage: $0 INPUT_DIR MAX_FILES_PER_JOB OUTPUT_PARENT_DIR [ARRAY_MAX_CONCURRENCY]"
  echo "  INPUT_DIR: Directory containing .fasta or .fa files"
  echo "  MAX_FILES_PER_JOB: Maximum number of .fasta files per array job"
  echo "  OUTPUT_PARENT_DIR: Parent directory for output chunks"
  echo "  ARRAY_MAX_CONCURRENCY: Maximum concurrent array tasks (default: 100)"
  exit 1
fi
if ! [[ "$MAX_FILES_PER_JOB" =~ ^[0-9]+$ && "$MAX_FILES_PER_JOB" -ge 1 ]]; then
  echo "MAX_FILES_PER_JOB must be a positive integer"
  exit 1
fi

# Normalize to absolute paths
INPUT_DIR="$(realpath -m "$INPUT_DIR")"
OUTPUT_PARENT_DIR="$(realpath -m "$OUTPUT_PARENT_DIR")"

# Make timestamped output directory that will also hold logs + manifest
TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_PARENT_DIR}/chunks_${TS}"
mkdir -p "$OUTPUT_DIR"

echo "Writing chunk files to: $OUTPUT_DIR"

# Collect .fasta and .fa files (absolute paths), null-safe & sorted
mapfile -d '' -t files < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -name "*.fasta" -o -name "*.fa" \) -print0 | sort -z)
total=${#files[@]}
if ((total == 0)); then
  echo "No .fasta or .fa files found in $INPUT_DIR"
  exit 1
fi

# Calculate number of chunks needed (ceil division)
NUM_CHUNKS=$(((total + MAX_FILES_PER_JOB - 1) / MAX_FILES_PER_JOB))

echo "Creating ${NUM_CHUNKS} chunk files (max ${MAX_FILES_PER_JOB} files per chunk)..."

# Create chunks with at most MAX_FILES_PER_JOB files each
for ((i = 0; i < NUM_CHUNKS; i++)); do
  start=$((i * MAX_FILES_PER_JOB))
  end=$((start + MAX_FILES_PER_JOB))
  ((end > total)) && end=$total
  ((start >= end)) && continue # skip empty chunk

  out="${OUTPUT_DIR}/id_${i}.txt"
  : >"$out"
  for ((j = start; j < end; j++)); do
    # Write absolute paths
    printf '%s\n' "${files[j]}" >>"$out"
  done
  echo "Wrote $(wc -l <"$out") paths -> $out"
done

# -------- Create total_paths.txt and processed_paths.txt --------
echo "Creating total_paths.txt and processed_paths.txt..."

TOTAL_PATHS_FILE="${OUTPUT_DIR}/total_paths.txt"
PROCESSED_PATHS_FILE="${OUTPUT_DIR}/processed_paths.txt"

# Collect all paths from all chunk files into total_paths.txt, sorted
: >"$TOTAL_PATHS_FILE"
while IFS= read -r -d '' f; do
  [[ -s "$f" ]] && cat "$f" >>"$TOTAL_PATHS_FILE"
done < <(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'id_*.txt' -print0 | sort -z)
sort -u "$TOTAL_PATHS_FILE" -o "$TOTAL_PATHS_FILE"

# Create empty processed_paths.txt
: >"$PROCESSED_PATHS_FILE"

echo "Created $TOTAL_PATHS_FILE with $(wc -l <"$TOTAL_PATHS_FILE") total paths"
echo "Created empty $PROCESSED_PATHS_FILE"

# -------- Build manifest (stable order) & submit array --------
echo "Building manifest and submitting array..."

MANIFEST="${OUTPUT_DIR}/filelist.manifest"
: >"$MANIFEST"

# Deterministic: sort by name; include only non-empty chunk files
while IFS= read -r -d '' f; do
  [[ -s "$f" ]] && realpath -s "$f" >>"$MANIFEST"
done < <(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'id_*.txt' -print0 | sort -z)

NUM_TASKS=$(wc -l <"$MANIFEST")
if ((NUM_TASKS == 0)); then
  echo "No non-empty id_*.txt chunk files found; nothing to submit."
  exit 0
fi

echo "Submitting ${NUM_TASKS} array tasks (max concurrent: ${ARRAY_MAX_CONCURRENCY})..."

# Get script directory to find run_msa_array.slrm
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSA_SCRIPT="${SCRIPT_DIR}/run_msa_array.slrm"

if [[ ! -f "$MSA_SCRIPT" ]]; then
  echo "ERROR: MSA script not found at $MSA_SCRIPT"
  exit 1
fi

# --parsable returns just the job ID so we can print it nicely
# Partition/account from config (run.sh exports SLURM_MSA_PARTITION, SLURM_ACCOUNT)
SLURM_OUTPUT="${SLURM_LOG_DIR:-/tmp}/%x.%A_%a.out"
SBATCH_OPTS=()
[[ -n "${SLURM_MSA_PARTITION:-}" ]] && SBATCH_OPTS+=(-p "$SLURM_MSA_PARTITION")
[[ -n "${SLURM_ACCOUNT:-}" ]] && SBATCH_OPTS+=(--account "$SLURM_ACCOUNT")
ARRAY_JOB_ID="$(
  sbatch --parsable \
    "${SBATCH_OPTS[@]}" \
    -o "$SLURM_OUTPUT" \
    --array=1-"$NUM_TASKS"%${ARRAY_MAX_CONCURRENCY} \
    --export=ALL,MANIFEST="$MANIFEST",BASE_OUTPUT_DIR="$OUTPUT_DIR",ORIGINAL_FASTA_DIR="$INPUT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
    "$MSA_SCRIPT"
)"

echo "Submitted array job ${ARRAY_JOB_ID} with ${NUM_TASKS} tasks."
echo "Chunks dir: $OUTPUT_DIR"
echo "Manifest:   $MANIFEST"
echo "MSA_ARRAY_JOB_ID=$ARRAY_JOB_ID"
echo "MSA_OUTPUT_DIR=$OUTPUT_DIR"
