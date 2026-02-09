#!/bin/bash
set -euo pipefail

# Usage: ./organize_boltz_outputs.sh BASE_OUTPUT_DIR [BOLTZ_CHUNKS_DIR] [CONFIG_FILE]
# Example: ./organize_boltz_outputs.sh /data/chunks_20240101_120000 /data/chunks_20240101_120000/boltz_chunks_20240101_120000
#          ./organize_boltz_outputs.sh /data/chunks_20240101_120000  (finds all boltz_chunks_* directories)

BASE_OUTPUT_DIR="${1:-}"
BOLTZ_CHUNKS_DIR="${2:-}"
CONFIG_FILE="${3:-}"

if [[ -z "${BASE_OUTPUT_DIR}" ]]; then
  echo "Usage: $0 BASE_OUTPUT_DIR [BOLTZ_CHUNKS_DIR] [CONFIG_FILE]"
  echo "  BASE_OUTPUT_DIR: Base output directory containing sequence directories (seq_idx/)"
  echo "  BOLTZ_CHUNKS_DIR: (Optional) Specific boltz_chunks directory. If not provided, processes all boltz_chunks_* directories"
  echo "  CONFIG_FILE: (Optional) Path to config.yaml. If not provided, uses project root config.yaml"
  exit 1
fi

# Get script directory to find parse_config.py and default config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If CONFIG_FILE not provided, use project root config.yaml
if [[ -z "${CONFIG_FILE}" ]]; then
  CONFIG_FILE="$(dirname "$SCRIPT_DIR")/config.yaml"
fi

# Check if delete_msa_after_processing is enabled
DELETE_MSA=false
if [[ -f "$CONFIG_FILE" ]]; then
  DELETE_MSA_OPTION=$(python3 "${SCRIPT_DIR}/parse_config.py" "$CONFIG_FILE" "boltz.delete_msa_after_processing" 2>/dev/null || echo "false")
  if [[ "$DELETE_MSA_OPTION" == "True" ]] || [[ "$DELETE_MSA_OPTION" == "true" ]] || [[ "$DELETE_MSA_OPTION" == "1" ]]; then
    DELETE_MSA=true
  fi
fi

if [[ "$DELETE_MSA" == "true" ]]; then
  echo "MSA deletion enabled: MSA files will be deleted after successful boltz processing"
fi

# Normalize to absolute paths
BASE_OUTPUT_DIR="$(realpath -m "$BASE_OUTPUT_DIR")"

# If BOLTZ_CHUNKS_DIR not provided, find all boltz_chunks directories
if [[ -z "${BOLTZ_CHUNKS_DIR}" ]]; then
  mapfile -d '' boltz_chunks_dirs < <(find "$BASE_OUTPUT_DIR" -maxdepth 1 -name "*boltz_chunk*" -type d -print0 | sort -z)
  if (( ${#boltz_chunks_dirs[@]} == 0 )); then
    echo "WARNING: No boltz_chunks directories found in $BASE_OUTPUT_DIR"
    exit 0
  fi
  echo "Found ${#boltz_chunks_dirs[@]} boltz_chunks directory(ies), processing all..."
else
  BOLTZ_CHUNKS_DIR="$(realpath -m "$BOLTZ_CHUNKS_DIR")"
  boltz_chunks_dirs=("$BOLTZ_CHUNKS_DIR")
fi

TOTAL_PROCESSED=0
TOTAL_SKIPPED=0
TOTAL_MSA_DELETED=0
TOTAL_ERRORS=0

# Process each boltz_chunks directory
for BOLTZ_CHUNKS_DIR in "${boltz_chunks_dirs[@]}"; do
  BOLTZ_OUTPUT_BASE="${BOLTZ_CHUNKS_DIR}/boltz_output"
  
  if [[ ! -d "$BOLTZ_OUTPUT_BASE" ]]; then
    echo "WARNING: Boltz output directory not found: $BOLTZ_OUTPUT_BASE, skipping..."
    continue
  fi
  
  echo "==============================================="
  echo "Organizing boltz outputs"
  echo "  Base dir: $BASE_OUTPUT_DIR"
  echo "  Boltz chunks dir: $BOLTZ_CHUNKS_DIR"
  echo "  Boltz output: $BOLTZ_OUTPUT_BASE"
  echo "==============================================="
  
  # Find all prediction directories
  # Pattern: boltz_output/chunk_X/boltz_results_chunk_X/predictions/seq_idx/
  mapfile -d '' prediction_dirs < <(find "$BOLTZ_OUTPUT_BASE" -type d -path "*/predictions/*" -print0 | sort -z)

  if (( ${#prediction_dirs[@]} == 0 )); then
    echo "No prediction directories found in $BOLTZ_OUTPUT_BASE, skipping..."
    continue
  fi

  echo "Found ${#prediction_dirs[@]} prediction directories"

  PROCESSED_COUNT=0
  SKIPPED_COUNT=0
  MSA_DELETED_COUNT=0
  ERROR_COUNT=0
  
  # Files for tracking organized sequences and processed paths
  PROCESSED_PATHS_FILE="${BOLTZ_CHUNKS_DIR}/processed_paths.txt"
  ORGANIZED_SEQ_FILE="${BOLTZ_CHUNKS_DIR}/organized_sequences.txt"
  : > "$ORGANIZED_SEQ_FILE"

for pred_dir in "${prediction_dirs[@]}"; do
  # Extract sequence ID from directory name (handle both seq_idx and idx formats)
  PRED_DIR_NAME=$(basename "$pred_dir")
  
  # Extract numeric ID (remove 'seq_' prefix if present)
  if [[ "$PRED_DIR_NAME" =~ ^seq_ ]]; then
    SEQ_ID="${PRED_DIR_NAME#seq_}"
  else
    SEQ_ID="$PRED_DIR_NAME"
  fi
  
  # Skip if SEQ_ID is empty or invalid
  if [[ -z "$SEQ_ID" ]]; then
    echo "WARNING: Could not extract sequence ID from: $pred_dir"
    ((SKIPPED_COUNT++)) || true
    continue
  fi
  
  # Always use seq_ format for target directory to match existing structure
  # This ensures boltz outputs go into seq_idx/boltz/ matching the existing seq_idx/msa/ structure
  TARGET_DIR="${BASE_OUTPUT_DIR}/seq_${SEQ_ID}/boltz"
  mkdir -p "$TARGET_DIR"
  
  # Find all model files and determine the highest model number
  # Pattern: *_model_N.* where N is the model number
  HIGHEST_MODEL=-1
  
  while IFS= read -r -d '' file; do
    # Extract model number from filename (e.g., seq_34073_model_9.cif -> 9)
    if [[ "$(basename "$file")" =~ _model_([0-9]+)\. ]]; then
      MODEL_NUM="${BASH_REMATCH[1]}"
      if [[ "$MODEL_NUM" =~ ^[0-9]+$ ]] && (( MODEL_NUM > HIGHEST_MODEL )); then
        HIGHEST_MODEL=$MODEL_NUM
      fi
    fi
  done < <(find "$pred_dir" -type f -print0)
  
  if (( HIGHEST_MODEL < 0 )); then
    echo "WARNING: No model files found in $pred_dir"
    ((SKIPPED_COUNT++)) || true
    continue
  fi
  
  # Copy only files for the highest model number
  FILES_TO_COPY=()
  while IFS= read -r -d '' file; do
    if [[ "$(basename "$file")" =~ _model_${HIGHEST_MODEL}\. ]]; then
      FILES_TO_COPY+=("$file")
    fi
  done < <(find "$pred_dir" -type f -print0)
  
  if (( ${#FILES_TO_COPY[@]} == 0 )); then
    echo "WARNING: No files to copy for $SEQ_ID (model ${HIGHEST_MODEL})"
    ((SKIPPED_COUNT++)) || true
    continue
  fi
  
  # Copy files with error handling
  COPY_FAILED=false
  COPIED_FILES=()
  for file in "${FILES_TO_COPY[@]}"; do
    DEST_FILE="${TARGET_DIR}/$(basename "$file")"
    if cp "$file" "$DEST_FILE"; then
      COPIED_FILES+=("$DEST_FILE")
    else
      echo "ERROR: Failed to copy $file to $DEST_FILE"
      COPY_FAILED=true
      ((ERROR_COUNT++)) || true
    fi
  done
  
  if [[ "$COPY_FAILED" == "true" ]]; then
    echo "ERROR: Some files failed to copy for $SEQ_ID, skipping cleanup"
    ((ERROR_COUNT++)) || true
    continue
  fi
  
  # Verify all copied files exist in target before proceeding
  VERIFICATION_FAILED=false
  for dest_file in "${COPIED_FILES[@]}"; do
    if [[ ! -f "$dest_file" ]]; then
      echo "ERROR: Copied file not found in target: $dest_file"
      VERIFICATION_FAILED=true
    fi
  done
  
  if [[ "$VERIFICATION_FAILED" == "true" ]]; then
    echo "ERROR: Verification failed for $SEQ_ID, skipping cleanup and tracking"
    ((ERROR_COUNT++)) || true
    continue
  fi
  
  FILES_COPIED=${#COPIED_FILES[@]}
  echo "Organized $SEQ_ID: copied ${FILES_COPIED} files for model ${HIGHEST_MODEL} -> $TARGET_DIR"
  ((PROCESSED_COUNT++)) || true
  
  # Track organized sequence
  echo "seq_${SEQ_ID}" >> "$ORGANIZED_SEQ_FILE"
  
  # Update processed_paths.txt with corresponding yaml file paths
  # Find yaml files that correspond to this sequence from tot_filesboltz.txt
  TOT_FILES_BOLTZ="${BOLTZ_CHUNKS_DIR}/tot_filesboltz.txt"
  if [[ -f "$TOT_FILES_BOLTZ" ]]; then
    # Use file locking for concurrent access
    LOCK_FILE="${PROCESSED_PATHS_FILE}.lock"
    if (
      flock -x -n 200
      grep "seq_${SEQ_ID}" "$TOT_FILES_BOLTZ" | while IFS= read -r yaml_path; do
        if [[ -f "$yaml_path" ]]; then
          # Check if already in processed_paths.txt to avoid duplicates
          if ! grep -Fxq "$yaml_path" "$PROCESSED_PATHS_FILE" 2>/dev/null; then
            echo "$yaml_path" >> "$PROCESSED_PATHS_FILE"
          fi
        fi
      done
    ) 200>"$LOCK_FILE" 2>/dev/null; then
      rm -f "$LOCK_FILE"
    else
      echo "  WARNING: Could not update processed_paths.txt (lock unavailable)"
    fi
  fi
  
  # Cleanup: delete all files from source directory after successful copy and verification
  FILES_DELETED=0
  while IFS= read -r -d '' file; do
    rm -f "$file"
    ((FILES_DELETED++)) || true
  done < <(find "$pred_dir" -type f -print0)
  
  if (( FILES_DELETED > 0 )); then
    echo "  Cleaned up ${FILES_DELETED} files from source directory"
  fi
  
  # Delete MSA directory if enabled
  if [[ "$DELETE_MSA" == "true" ]]; then
    MSA_DIR="${BASE_OUTPUT_DIR}/seq_${SEQ_ID}/msa"
    if [[ -d "$MSA_DIR" ]]; then
      MSA_FILES_COUNT=$(find "$MSA_DIR" -type f | wc -l)
      rm -rf "$MSA_DIR"
      echo "  Deleted MSA directory ($MSA_FILES_COUNT files) for $SEQ_ID"
      ((MSA_DELETED_COUNT++)) || true
    fi
  fi
done

  echo "==============================================="
  echo "Organization complete for $BOLTZ_CHUNKS_DIR"
  echo "  Processed: $PROCESSED_COUNT sequences"
  echo "  Skipped: $SKIPPED_COUNT sequences"
  echo "  Errors: $ERROR_COUNT sequences"
  if [[ "$DELETE_MSA" == "true" ]]; then
    echo "  MSA directories deleted: $MSA_DELETED_COUNT"
  fi
  if [[ -f "$ORGANIZED_SEQ_FILE" ]]; then
    echo "  Organized sequences tracked in: $ORGANIZED_SEQ_FILE"
  fi
  echo "==============================================="
  
  TOTAL_PROCESSED=$((TOTAL_PROCESSED + PROCESSED_COUNT))
  TOTAL_SKIPPED=$((TOTAL_SKIPPED + SKIPPED_COUNT))
  TOTAL_MSA_DELETED=$((TOTAL_MSA_DELETED + MSA_DELETED_COUNT))
  TOTAL_ERRORS=$((TOTAL_ERRORS + ERROR_COUNT))
done

echo "==============================================="
echo "Overall organization complete"
echo "  Total processed: $TOTAL_PROCESSED sequences"
echo "  Total skipped: $TOTAL_SKIPPED sequences"
echo "  Total errors: $TOTAL_ERRORS sequences"
if [[ "$DELETE_MSA" == "true" ]]; then
  echo "  Total MSA directories deleted: $TOTAL_MSA_DELETED"
fi
echo "==============================================="

# Exit with error code if there were any errors
if (( TOTAL_ERRORS > 0 )); then
  echo "WARNING: Organization completed with $TOTAL_ERRORS errors"
  exit 1
fi

