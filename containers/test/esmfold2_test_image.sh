#!/usr/bin/env bash

set -euo pipefail

IMAGE_PATH="${1}"
CACHE_DIR="${2}"

echo "Launchin test for ESMFOLD2..."

singularity exec --nv \
  -B $CACHE_DIR:$CACHE_DIR \
  $IMAGE_PATH python /opt/run_esmfold2.py --cache $CACHE_DIR

echo "Run completed."
