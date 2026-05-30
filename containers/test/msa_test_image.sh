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

THREADS=${SLURM_CPUS_PER_TASK:-16}
INPUT_FASTA=/n/home06/tbush/original.fasta
OUTPUT_DIR=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/singularity_dev/images/output_tests

#image MSA
MSA_IMAGE=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/singularity_dev/images/msa.sif

MSA_DB=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/singularity_dev/images/output_tests

echo "Running singularity image at ${MSA_IMAGE}..."

singularity exec -nv \
  -B $INPUT_FASTA:$INPUT_FASTA \
  -B $OUTPUT_DIR:$OUTPUT_DIR \
  -B $MSA_IMAGE:$MSA_IMAGE \
  $MSA_IMAGE colabfold_search "$INPUT_FASTA" "$MSA_DB" "$OUTPUT_DIR" --gpu 1 --thread "$THREADS"
