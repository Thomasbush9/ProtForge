#!/bin/bash
# ESMFold prototype launcher. No config parsing - set everything via env vars.
#
# Example:
#   FASTA_DIR=~/tmp_fastas \
#   OUTPUT_DIR=/n/holylfs06/.../esmfold_out \
#   ESMFOLD_ENV_PREFIX=/n/home06/USER/envs/esmfold \
#   ESMFOLD_CACHE_DIR=/n/holylfs06/.../esmfold_cache \
#   PARTITION=kempner_requeue \
#   ACCOUNT=kempner_foo_lab \
#   LOG_DIR=/n/home06/USER/job_logs \
#   bash esmfold_skeleton/launch.sh
#
# Pre-req: run `python esmfold_skeleton/download_esmfold.py --cache-dir $ESMFOLD_CACHE_DIR`
# on a login node first to populate the HF cache.

set -euo pipefail

: "${FASTA_DIR:?Set FASTA_DIR}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${ESMFOLD_ENV_PREFIX:?Set ESMFOLD_ENV_PREFIX}"
: "${ESMFOLD_CACHE_DIR:?Set ESMFOLD_CACHE_DIR}"
: "${PARTITION:?Set PARTITION}"
: "${ACCOUNT:?Set ACCOUNT}"
: "${LOG_DIR:?Set LOG_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESMFOLD_SCRIPT="${SCRIPT_DIR}/run_esmfold.py"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

sbatch \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --output="${LOG_DIR}/esmfold.%j.out" \
    --export=ALL,FASTA_DIR="$FASTA_DIR",OUTPUT_DIR="$OUTPUT_DIR",ESMFOLD_ENV_PREFIX="$ESMFOLD_ENV_PREFIX",ESMFOLD_CACHE_DIR="$ESMFOLD_CACHE_DIR",ESMFOLD_SCRIPT="$ESMFOLD_SCRIPT" \
    "${SCRIPT_DIR}/run_esmfold.slrm"
