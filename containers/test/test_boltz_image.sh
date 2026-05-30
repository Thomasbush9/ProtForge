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

# define .yaml dir, msa dir, output dir
INPUT_YAML="${1}"
INPUT_MSA="${2}"
OUTPUT_DIR="${3}"

#give path to image
BOLTZ_IMAGE="${4}"
#define cache path for boltz weights:
CACHE_PATH=/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db

echo "Running singularity Image at ${BOLTZ_IMAGE}"

# run the image:
singularity exec --nv \
  -B $INPUT_YAML:$INPUT_YAML \
  -B $INPUT_MSA:$INPUT_MSA \
  -B $OUTPUT_DIR:$OUTPUT_DIR \
  -B $CACHE_PATH:/weights \
  $BOLTZ_IMAGE boltz predict $INPUT_YAML --cache /weights --out_dir $OUTPUT_DIR
