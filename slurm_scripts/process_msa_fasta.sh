#!/bin/bash
set -euo pipefail

# Usage: ./process_msa_fasta.sh ORIGINAL_FASTA_DIR OUTPUT_DIR
# Example: ./process_msa_fasta.sh /data/fasta /data/chunks_20240101_120000
# Converts .fasta files to .yaml format with MSA paths

ORIGINAL_FASTA_DIR="${1:-}"
OUTPUT_DIR="${2:-}"

if [[ -z "${ORIGINAL_FASTA_DIR}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: $0 ORIGINAL_FASTA_DIR OUTPUT_DIR"
  echo "  ORIGINAL_FASTA_DIR: Directory containing original .fasta files"
  echo "  OUTPUT_DIR: Directory containing sequence folders (seqnumber/msa/)"
  echo "  Output: Creates .yaml files in OUTPUT_DIR/seqnumber/"
  exit 1
fi

# Normalize to absolute paths
ORIGINAL_FASTA_DIR="$(realpath -m "$ORIGINAL_FASTA_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

PROCESSED_COUNT=0
SKIPPED_COUNT=0

echo "Processing FASTA files..."
echo "Original FASTA dir: $ORIGINAL_FASTA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "==============================================="

# Process each .fasta or .fa file in the original directory
while IFS= read -r -d '' INPUT_FASTA; do
  BASENAME=$(basename "$INPUT_FASTA")
  BASENAME_NOEXT="${BASENAME%.fa}"
  BASENAME_NOEXT="${BASENAME_NOEXT%.fasta}"

  # Find corresponding a3m file in OUTPUT_DIR/seqnumber/msa/
  SEQ_DIR="${OUTPUT_DIR}/${BASENAME_NOEXT}"
  A3M_FILE=$(find "$SEQ_DIR/msa" -name "*.a3m" -type f 2>/dev/null | head -n1)

  if [[ -z "$A3M_FILE" || ! -f "$A3M_FILE" ]]; then
    echo "WARNING: No a3m file found for $BASENAME in $SEQ_DIR/msa/, skipping..."
    ((SKIPPED_COUNT++)) || true
    continue
  fi

  # Get absolute path to a3m file
  A3M_ABSOLUTE=$(realpath "$A3M_FILE")
  OUTPUT_YAML="${SEQ_DIR}/${BASENAME_NOEXT}.yaml"

  # Read original header and sequence
  ORIGINAL_HEADER=$(head -n1 "$INPUT_FASTA")
  SEQUENCE=$(tail -n+2 "$INPUT_FASTA" | tr -d '\n')

  # Extract ID from header (remove '>' and take first field, or use basename)
  PROTEIN_ID="${ORIGINAL_HEADER#>}"
  PROTEIN_ID="${PROTEIN_ID%% *}"
  PROTEIN_ID="${PROTEIN_ID%%|*}"
  if [[ -z "$PROTEIN_ID" ]]; then
    PROTEIN_ID="$BASENAME_NOEXT"
  fi

  # Write YAML file (quote id to ensure it's always a string)
  {
    echo "version: 1"
    echo ""
    echo "sequences:"
    echo "  - protein:"
    echo "      id: \"$PROTEIN_ID\""
    echo "      sequence: $SEQUENCE"
    echo "      msa: $A3M_ABSOLUTE"
  } >"$OUTPUT_YAML"

  echo "Processed: $BASENAME -> $OUTPUT_YAML"
  echo "  ID: $PROTEIN_ID"
  echo "  MSA: $A3M_ABSOLUTE"
  ((PROCESSED_COUNT++)) || true

done < <(find "$ORIGINAL_FASTA_DIR" -maxdepth 1 -type f \( -name "*.fasta" -o -name "*.fa" \) -print0)
#TODO add here the script to add the multiple sequences if specificed
echo "==============================================="
echo "Processing complete"
echo "  Processed: $PROCESSED_COUNT"
echo "  Skipped: $SKIPPED_COUNT"
echo "==============================================="
