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

# find most recent chunks directory in the output directory (by modification time)
chunks_dir=$(find "$ROOT_DIR" -maxdepth 1 -type d -name "chunks_*" -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)

MSA_CHECK_SKIP=false
if [ -z "$chunks_dir" ] || [ ! -d "$chunks_dir" ]; then
    echo "Warning: No chunks directory found in $ROOT_DIR"
    echo "  Skipping MSA checking"
    MSA_CHECK_SKIP=true
else
    MSA_CHUNKS_DIR="$chunks_dir"
    echo "Using MSA chunks directory: $MSA_CHUNKS_DIR"
fi

# Check MSA outputs if chunks directory exists
if [ "$MSA_CHECK_SKIP" = false ]; then
    # get paths to files not processed
    TOTAL_PATHS_FILE=$MSA_CHUNKS_DIR/total_paths.txt
    PROCESSED_PATHS_FILE=$MSA_CHUNKS_DIR/processed_paths.txt

    if [ ! -f "$TOTAL_PATHS_FILE" ]; then
        echo "Error: total_paths.txt not found in $MSA_CHUNKS_DIR"
        exit 1
    fi

    if [ ! -f "$PROCESSED_PATHS_FILE" ]; then
        echo "Error: processed_paths.txt not found in $MSA_CHUNKS_DIR"
        exit 1
    fi

    # find unprocessed paths using comm
    UNPROCESSED_PATHS_FILE=$ROOT_DIR/msa_unprocessed_paths.txt
    comm -23 <(sort -u "$TOTAL_PATHS_FILE") <(sort -u "$PROCESSED_PATHS_FILE") > "$UNPROCESSED_PATHS_FILE"

    # get number of unprocessed paths
    NUM_UNPROCESSED_PATHS=$(wc -l < "$UNPROCESSED_PATHS_FILE")

    # print number of unprocessed paths
    echo "Number of unprocessed MSA paths: $NUM_UNPROCESSED_PATHS"

    # print unprocessed paths
    if [ "$NUM_UNPROCESSED_PATHS" -gt 0 ]; then
        cat "$UNPROCESSED_PATHS_FILE"
    fi

    # Launch retry workflow for unprocessed paths if any exist
    if [ "$NUM_UNPROCESSED_PATHS" -gt 0 ]; then
        echo "==============================================="
        echo "Launching MSA retry for unprocessed paths"
        echo "==============================================="

        # Get script directory
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        CONFIG_FILE="${CONFIG_FILE:-$(dirname "$SCRIPT_DIR")/config.yaml}"

        # Read configuration
        if [ ! -f "$CONFIG_FILE" ]; then
            echo "Error: Config file not found at $CONFIG_FILE"
            exit 1
        fi

        MAX_FILES_PER_JOB=$(python3 "${SCRIPT_DIR}/parse_config.py" "$CONFIG_FILE" "msa.max_files_per_job")
        ARRAY_MAX_CONCURRENCY=$(python3 "${SCRIPT_DIR}/parse_config.py" "$CONFIG_FILE" "msa.array_max_concurrency")
        OUTPUT_PARENT_DIR=$(python3 "${SCRIPT_DIR}/parse_config.py" "$CONFIG_FILE" "output.parent_dir")

        if [ -z "$MAX_FILES_PER_JOB" ] || [ -z "$ARRAY_MAX_CONCURRENCY" ] || [ -z "$OUTPUT_PARENT_DIR" ]; then
            echo "Error: Failed to read MSA configuration from $CONFIG_FILE"
            exit 1
        fi

        echo "Max files per job: $MAX_FILES_PER_JOB"
        echo "Array max concurrency: $ARRAY_MAX_CONCURRENCY"
        echo "Output parent directory: $OUTPUT_PARENT_DIR"

        # Create new timestamped chunks directory
        TS="$(date +%Y%m%d_%H%M%S)"
        NEW_MSA_CHUNKS_DIR="${ROOT_DIR}/chunks_retry_${TS}"
        mkdir -p "$NEW_MSA_CHUNKS_DIR"

        echo "Creating new MSA chunks directory: $NEW_MSA_CHUNKS_DIR"

        # Read unprocessed paths into array
        mapfile -t unprocessed_paths < "$UNPROCESSED_PATHS_FILE"
        total=${#unprocessed_paths[@]}

        # Calculate number of chunks needed
        NUM_CHUNKS=$(( (total + MAX_FILES_PER_JOB - 1) / MAX_FILES_PER_JOB ))

        echo "Creating ${NUM_CHUNKS} chunk files (max ${MAX_FILES_PER_JOB} files per chunk)..."

        # Create chunk files with unprocessed paths
        for ((i=0; i<NUM_CHUNKS; i++)); do
            start=$(( i * MAX_FILES_PER_JOB ))
            end=$(( start + MAX_FILES_PER_JOB ))
            (( end > total )) && end=$total
            (( start >= end )) && continue

            out="${NEW_MSA_CHUNKS_DIR}/id_${i}.txt"
            : > "$out"
            for ((j=start; j<end; j++)); do
                path="${unprocessed_paths[j]}"
                path="${path//$'\r'/}"  # Remove carriage returns
                if [ -f "$path" ]; then
                    printf '%s\n' "$path" >> "$out"
                fi
            done

            file_count=$(wc -l < "$out" | tr -d ' ')
            if [ "$file_count" -gt 0 ]; then
                echo "Created id_${i}.txt with ${file_count} paths"
            fi
        done

        # Create total_paths.txt and processed_paths.txt for retry
        echo "Creating total_paths.txt and processed_paths.txt..."

        TOTAL_PATHS_FILE_NEW="${NEW_MSA_CHUNKS_DIR}/total_paths.txt"
        PROCESSED_PATHS_FILE_NEW="${NEW_MSA_CHUNKS_DIR}/processed_paths.txt"

        # Write all unprocessed paths to total_paths.txt
        : > "$TOTAL_PATHS_FILE_NEW"
        for path in "${unprocessed_paths[@]}"; do
            path="${path//$'\r'/}"  # Remove carriage returns
            echo "$path" >> "$TOTAL_PATHS_FILE_NEW"
        done
        sort -u "$TOTAL_PATHS_FILE_NEW" -o "$TOTAL_PATHS_FILE_NEW"

        # Create empty processed_paths.txt
        : > "$PROCESSED_PATHS_FILE_NEW"

        echo "Created $TOTAL_PATHS_FILE_NEW with $(wc -l < "$TOTAL_PATHS_FILE_NEW") total paths"
        echo "Created empty $PROCESSED_PATHS_FILE_NEW"

        # Build manifest and submit array job
        echo "Building manifest and submitting array job..."

        MANIFEST="${NEW_MSA_CHUNKS_DIR}/filelist.manifest"
        : > "$MANIFEST"

        # Include only non-empty chunk files
        while IFS= read -r -d '' f; do
            [[ -s "$f" ]] && realpath -s "$f" >> "$MANIFEST"
        done < <(find "$NEW_MSA_CHUNKS_DIR" -maxdepth 1 -type f -name 'id_*.txt' -print0 | sort -z)

        NUM_TASKS=$(wc -l < "$MANIFEST")
        if (( NUM_TASKS == 0 )); then
            echo "No non-empty chunk files found; nothing to submit."
        else
            echo "Submitting ${NUM_TASKS} array tasks (max concurrent: ${ARRAY_MAX_CONCURRENCY})..."

            MSA_SCRIPT="${SCRIPT_DIR}/run_msa_array.slrm"
            if [[ ! -f "$MSA_SCRIPT" ]]; then
                echo "ERROR: MSA array script not found at $MSA_SCRIPT"
            else
                # Find original FASTA directory from first unprocessed path
                # Extract directory path (assuming all paths are from same directory)
                ORIGINAL_FASTA_DIR=$(dirname "${unprocessed_paths[0]}")
                ORIGINAL_FASTA_DIR="${ORIGINAL_FASTA_DIR//$'\r'/}"  # Remove carriage returns

                # Submit array job (inherit MMSEQ2_DB, COLABFOLD_DB, etc. via --export=ALL from checker.sh)
                SLURM_OUTPUT="${SLURM_LOG_DIR:-/tmp}/%x.%A_%a.out"
                RETRY_ARRAY_JOB_ID="$(
                    sbatch --parsable \
                        -o "$SLURM_OUTPUT" \
                        --array=1-"$NUM_TASKS"%${ARRAY_MAX_CONCURRENCY} \
                        --export=ALL,MANIFEST="$MANIFEST",BASE_OUTPUT_DIR="$ROOT_DIR",ORIGINAL_FASTA_DIR="$ORIGINAL_FASTA_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
                        "$MSA_SCRIPT"
                )"

                echo "Submitted retry array job ${RETRY_ARRAY_JOB_ID} with ${NUM_TASKS} tasks."
                echo "Chunks dir: $NEW_MSA_CHUNKS_DIR"
                echo "Manifest:   $MANIFEST"

                # Submit post-processing job after retry completes
                # Post-processing will merge results into original chunks directory structure
                # For simplicity, we'll run post-processing on the original output directory
                # using the original FASTA directory
                echo ""
                echo "Submitting post-processing job (FASTA->YAML conversion)..."
                
                # Create a script that will wait for array job and run post-processing
                # We submit a simple job that waits for the array and then runs post-processing
                PROCESS_SCRIPT="${SCRIPT_DIR}/process_msa_fasta.sh"
                if [[ -f "$PROCESS_SCRIPT" ]]; then
                    # Merge processed_paths from retry into original
                    # Post-processing should run on the original output directory structure
                    # We'll submit a job that depends on the retry array job
                    POST_PROCESS_OUTPUT="${SLURM_LOG_DIR:-/tmp}/%x.%j.out"
                    POST_PROCESS_JOB_ID=$(sbatch --parsable \
                        -o "$POST_PROCESS_OUTPUT" \
                        --dependency=afterok:${RETRY_ARRAY_JOB_ID} \
                        --export=ALL,ORIGINAL_FASTA_DIR="$ORIGINAL_FASTA_DIR",OUTPUT_DIR="$ROOT_DIR",SCRIPT_DIR="$SCRIPT_DIR" \
                        --wrap="bash -c 'if [ -f \"$PROCESS_SCRIPT\" ] && [ -n \"\$ORIGINAL_FASTA_DIR\" ] && [ -n \"\$OUTPUT_DIR\" ]; then \"$PROCESS_SCRIPT\" \"\$ORIGINAL_FASTA_DIR\" \"\$OUTPUT_DIR\"; else echo \"Skipping post-processing: missing script or env vars\"; fi'" 2>/dev/null || echo "")
                    
                    if [[ -n "$POST_PROCESS_JOB_ID" ]]; then
                        echo "Submitted post-processing job ${POST_PROCESS_JOB_ID} (depends on retry array job ${RETRY_ARRAY_JOB_ID})"
                    else
                        echo "WARNING: Failed to submit post-processing job"
                    fi
                else
                    echo "WARNING: Post-processing script not found at $PROCESS_SCRIPT, skipping"
                fi
            fi
        fi

        echo "==============================================="
        echo "MSA retry workflow launched successfully"
        echo "==============================================="
    else
        echo "All MSA paths processed."
        
        # Even if all paths are processed, check if post-processing needs to run
        # This handles the case where MSA completed but post-processing failed
        POST_PROCESS_DONE="${ROOT_DIR}/post_process.done"
        if [ ! -f "$POST_PROCESS_DONE" ]; then
            echo ""
            echo "All MSA files processed, but post-processing may not have completed."
            echo "Running post-processing (FASTA->YAML conversion)..."
            
            SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
            PROCESS_SCRIPT="${SCRIPT_DIR}/process_msa_fasta.sh"
            
            # Try to determine original FASTA directory
            # Look for ORIGINAL_FASTA_DIR in the chunks directory or use first path
            ORIGINAL_FASTA_DIR=""
            if [ -f "$TOTAL_PATHS_FILE" ] && [ -s "$TOTAL_PATHS_FILE" ]; then
                FIRST_PATH=$(head -n1 "$TOTAL_PATHS_FILE")
                ORIGINAL_FASTA_DIR=$(dirname "$FIRST_PATH")
            fi
            
            if [[ -f "$PROCESS_SCRIPT" ]] && [[ -n "$ORIGINAL_FASTA_DIR" ]] && [[ -d "$ORIGINAL_FASTA_DIR" ]]; then
                if "$PROCESS_SCRIPT" "$ORIGINAL_FASTA_DIR" "$ROOT_DIR"; then
                    touch "$POST_PROCESS_DONE"
                    echo "Post-processing completed successfully"
                else
                    echo "WARNING: Post-processing failed"
                fi
            else
                echo "WARNING: Cannot run post-processing - missing script or FASTA directory"
                echo "  PROCESS_SCRIPT exists: $([ -f "$PROCESS_SCRIPT" ] && echo 'YES' || echo 'NO')"
                echo "  ORIGINAL_FASTA_DIR: ${ORIGINAL_FASTA_DIR:-'not found'}"
            fi
        else
            echo "Post-processing already completed (post_process.done exists)."
        fi
    fi
fi


