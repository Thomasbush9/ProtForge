#!/bin/bash
set -euo pipefail

OUTPUT_DIR="${1:-}"
N="${2:-}"
OUTPUT_PARENT_DIR="${3:-}"
ARRAY_MAX_CONCURRENCY="${4:-100}"

if [[ -z "${OUTPUT_DIR}" || -z "${N}" || -z "${OUTPUT_PARENT_DIR}" ]]; then
  echo "Usage: $0 OUTPUT_DIR N OUTPUT_PARENT_DIR [ARRAY_MAX_CONCURRENCY]"
  echo "  OUTPUT_DIR: Directory containing .yaml files (recursively searched)"
  echo "  N: Number of chunks for ESM processing"
  echo "  OUTPUT_PARENT_DIR: Parent directory for ESM chunk output"
  echo "  ARRAY_MAX_CONCURRENCY: Maximum concurrent array tasks (default: 100)"
  exit 1
fi
if ! [[ "$N" =~ ^[0-9]+$ && "$N" -ge 1 ]]; then
  echo "N must be a positive integer"
  exit 1
fi

# Normalize to absolute path
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

# Make timestamped output directory that will also hold logs + manifest
TS="$(date +%Y%m%d_%H%M%S)"
ESM_CHUNKS_DIR="${OUTPUT_PARENT_DIR}/esm_chunks_${TS}"
mkdir -p "$ESM_CHUNKS_DIR"

echo "Finding .yaml files in: $OUTPUT_DIR"
echo "Chunk files will be created in: $ESM_CHUNKS_DIR"

# Recursively collect all .yaml files (absolute paths), null-safe & sorted
# Use -L to follow symlinks (needed for checker_esm.sh retry workflow)
mapfile -d '' -t yaml_files < <(find -L "$OUTPUT_DIR" -type f -name "*.yaml" -print0 | sort -z)
total=${#yaml_files[@]}

if ((total == 0)); then
  echo "No .yaml files found in $OUTPUT_DIR"
  exit 1
fi

echo "Found ${total} .yaml files"

# Compute chunk size (ceil division)
chunk_size=$(((total + N - 1) / N))

# Split into N chunks (skip empties if N > total)
for ((i = 0; i < N; i++)); do
  start=$((i * chunk_size))
  end=$((start + chunk_size))
  ((end > total)) && end=$total
  ((start >= end)) && continue # skip empty chunk

  out="${ESM_CHUNKS_DIR}/id_${i}.txt"
  : >"$out"
  for ((j = start; j < end; j++)); do
    # Write absolute paths to yaml files
    printf '%s\n' "${yaml_files[j]}" >>"$out"
  done
  echo "Wrote $(wc -l <"$out") paths -> $out"
done

# -------- Build manifest (stable order) & submit array --------
echo "Building manifest and submitting array..."

MANIFEST="${ESM_CHUNKS_DIR}/filelist.manifest"
: >"$MANIFEST"

# Deterministic: sort by name; include only non-empty chunk files
while IFS= read -r -d '' f; do
  [[ -s "$f" ]] && realpath -s "$f" >>"$MANIFEST"
done < <(find "$ESM_CHUNKS_DIR" -maxdepth 1 -type f -name 'id_*.txt' -print0 | sort -z)

NUM_TASKS=$(wc -l <"$MANIFEST")
if ((NUM_TASKS == 0)); then
  echo "No non-empty id_*.txt chunk files found; nothing to submit."
  exit 0
fi
# write all the paths into total_paths.txt in the output directory
touch "${ESM_CHUNKS_DIR}/total_paths.txt"
for file in "${yaml_files[@]}"; do
  echo "$file" >>"${ESM_CHUNKS_DIR}/total_paths.txt"
done
# create a processed_paths.txt file in the output directory
touch "${ESM_CHUNKS_DIR}/processed_paths.txt"

echo "Submitting ${NUM_TASKS} array tasks (max concurrent: ${ARRAY_MAX_CONCURRENCY})..."

# Partition/account from config; dependency when set
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_OUTPUT="${SLURM_LOG_DIR:-/tmp}/%x.%A_%a.out"
SBATCH_OPTS=()
[[ -n "${SLURM_ESM_PARTITION:-}" ]] && SBATCH_OPTS+=(-p "$SLURM_ESM_PARTITION")
[[ -n "${SLURM_ACCOUNT:-}" ]] && SBATCH_OPTS+=(--account "$SLURM_ACCOUNT")
[[ -n "${DEPENDENCY_JOB_ID:-}" ]] && SBATCH_OPTS+=(--dependency=afterok:${DEPENDENCY_JOB_ID})
ARRAY_JOB_ID="$(
  sbatch --parsable \
    "${SBATCH_OPTS[@]}" \
    -o "$SLURM_OUTPUT" \
    --array=1-"$NUM_TASKS"%${ARRAY_MAX_CONCURRENCY} \
    --export=ALL,MANIFEST="$MANIFEST",BASE_OUTPUT_DIR="$OUTPUT_DIR",ESM_CHUNKS_DIR="$ESM_CHUNKS_DIR",ES_ENV_PREFIX="${ES_ENV_PREFIX:-}",ESM_WORK_DIR="${ESM_WORK_DIR:-}",ESM_CACHE_DIR="${ESM_CACHE_DIR:-}" \
    "${SCRIPT_DIR}/run_esm_array.slrm"
)"

echo "Submitted array job ${ARRAY_JOB_ID} with ${NUM_TASKS} tasks."
echo "Chunks dir: $ESM_CHUNKS_DIR"
echo "Manifest:   $MANIFEST"
