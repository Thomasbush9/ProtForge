#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
# Partition/account set by caller via sbatch flags (from config)
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --time=01:00:00
# Log dir from env (caller passes -o)
#SBATCH --output=/n/home06/tbush/job_logs/%x.%A_%a.out

set -euo pipefail

IMAGE_PATH="${1}"
CACHE_DIR="${2}"

echo "Launchin test for ESMFOLD2..."

singularity exec --nv \
  -B $CACHE_DIR:$CACHE_DIR \
  $IMAGE_PATH python /opt/run_esmfold2.py --cache $CACHE_DIR

echo "Run completed."
