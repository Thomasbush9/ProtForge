#!/bin/bash
set -euo pipefail

#enter the output directory
ROOT_DIR=$1

#check if the output directory exists
if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Output directory $ROOT_DIR does not exist."
    exit 1
fi

#check if the output directory is empty
if [ -z "$(ls -A $ROOT_DIR)" ]; then
    echo "Error: Output directory $ROOT_DIR is empty."
    exit 1
fi

# find most recent esm_chunks directory in the output directory (by modification time)
esm_chunks_dir=$(find "$ROOT_DIR" -maxdepth 1 -type d -name "esm_chunks_*" -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$esm_chunks_dir" ] || [ ! -d "$esm_chunks_dir" ]; then
    echo "Error: No esm_chunks directory found in $ROOT_DIR"
    exit 1
fi

ESM_CHUNKS_DIR="$esm_chunks_dir"
echo "Using ESM chunks directory: $ESM_CHUNKS_DIR"

# get paths to files
TOTAL_PATHS_FILE=$ESM_CHUNKS_DIR/total_paths.txt
PROCESSED_PATHS_FILE=$ESM_CHUNKS_DIR/processed_paths.txt

if [ ! -f "$TOTAL_PATHS_FILE" ]; then
    echo "Error: total_paths.txt not found in $ESM_CHUNKS_DIR"
    exit 1
fi

if [ ! -f "$PROCESSED_PATHS_FILE" ]; then
    echo "Error: processed_paths.txt not found in $ESM_CHUNKS_DIR"
    exit 1
fi

# find unprocessed paths using comm
UNPROCESSED_PATHS_FILE=$ESM_CHUNKS_DIR/esm_unprocessed_paths.txt
comm -23 <(sort -u "$TOTAL_PATHS_FILE") <(sort -u "$PROCESSED_PATHS_FILE") > "$UNPROCESSED_PATHS_FILE"

# get number of unprocessed paths
NUM_UNPROCESSED_PATHS=$(wc -l < "$UNPROCESSED_PATHS_FILE")

# print number of unprocessed paths
echo "Number of unprocessed paths: $NUM_UNPROCESSED_PATHS"

# print unprocessed paths
cat "$UNPROCESSED_PATHS_FILE"

# if no unprocessed paths, exit successfully
if [ "$NUM_UNPROCESSED_PATHS" -eq 0 ]; then
    echo "All paths processed. Exiting."
    exit 0
fi

# Launch retry workflow for unprocessed paths
echo "==============================================="
echo "Launching ESM retry for unprocessed paths"
echo "==============================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$(dirname "$SCRIPT_DIR")/config.yaml}"

# Read configuration
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

NUM_CHUNKS=$(python3 "${SCRIPT_DIR}/parse_config.py" "$CONFIG_FILE" "esm.num_chunks")
ARRAY_MAX_CONCURRENCY=$(python3 "${SCRIPT_DIR}/parse_config.py" "$CONFIG_FILE" "esm.array_max_concurrency")
OUTPUT_PARENT_DIR=$(python3 "${SCRIPT_DIR}/parse_config.py" "$CONFIG_FILE" "output.parent_dir")

if [ -z "$NUM_CHUNKS" ] || [ -z "$ARRAY_MAX_CONCURRENCY" ] || [ -z "$OUTPUT_PARENT_DIR" ]; then
    echo "Error: Failed to read ESM configuration from $CONFIG_FILE"
    exit 1
fi

echo "Number of chunks: $NUM_CHUNKS"
echo "Array max concurrency: $ARRAY_MAX_CONCURRENCY"
echo "Output parent directory: $OUTPUT_PARENT_DIR"

# Create temporary directory with symlinks to unprocessed yaml files
TEMP_DIR=$(mktemp -d -p "$ROOT_DIR" esm_retry_yamls_XXXXXX)
echo "Creating temporary directory for unprocessed yaml files: $TEMP_DIR"

# Read unprocessed paths and create symlinks with unique names
mapfile -t unprocessed_paths < "$UNPROCESSED_PATHS_FILE"
COUNTER=0
for path in "${unprocessed_paths[@]}"; do
    path="${path//$'\r'/}"  # Remove carriage returns
    if [ -f "$path" ]; then
        # Create unique symlink name to avoid conflicts from duplicate basenames
        # Use counter-based naming to ensure all symlinks have unique names
        BASENAME=$(basename "$path")
        # Extract stem and extension for counter-based naming
        if [[ "$BASENAME" == *.* ]]; then
            STEM="${BASENAME%.*}"
            EXT="${BASENAME##*.}"
            LINK_NAME="${STEM}_${COUNTER}.${EXT}"
        else
            LINK_NAME="${BASENAME}_${COUNTER}"
        fi
        ln -s "$path" "$TEMP_DIR/$LINK_NAME"
        ((COUNTER++)) || true
    else
        echo "Warning: File not found: $path"
    fi
done

echo "Created $(find "$TEMP_DIR" -type l | wc -l) symlinks in temporary directory"

# Call run_esm.sh with temp directory as OUTPUT_DIR
ESM_SCRIPT="${SCRIPT_DIR}/run_esm.sh"
if [[ ! -f "$ESM_SCRIPT" ]]; then
    echo "ERROR: ESM script not found at $ESM_SCRIPT"
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo "Launching ESM job with unprocessed paths..."
"$ESM_SCRIPT" "$TEMP_DIR" "$NUM_CHUNKS" "$OUTPUT_PARENT_DIR" "$ARRAY_MAX_CONCURRENCY"

echo "==============================================="
echo "ESM retry workflow launched successfully"
echo "==============================================="
echo "Note: Temporary directory $TEMP_DIR will remain for reference"
