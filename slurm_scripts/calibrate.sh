#!/usr/bin/env bash
# calibrate.sh — measure SLURM consumption for ProtForge stages on one GPU.
#
# Self-contained: writes its own config.yaml and does not read the webapp-
# managed config.yaml at the repo root. Shared Kempner paths are baked in
# from config.template.yaml; user-specific paths come from $USER (or env
# var overrides).
#
# Usage:
#   bash slurm_scripts/calibrate.sh <stage> <gpu_type> <fasta_dir> [output_dir]
#
# Example (full Option A sweep on h100):
#   bash slurm_scripts/calibrate.sh all h100 tests/calibration_inputs/fastas \
#       /n/holylfs06/LABS/bsabatini_lab/Everyone/$USER/calib_$(date +%Y%m%d_%H%M%S)
#
# After completion, $OUT_ROOT contains:
#   config.yaml          the rendered calibration config
#   logs/                slurm.log_dir
#   summary.txt          per-stage benchmark TSV listing
#   run/                 output.parent_dir (chunks/, sequences/, benchmarks/)

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 <stage> <gpu_type> <fasta_dir> [output_dir]

  stage      One of: msa, boltz, esm, esmfold, es, all
             'all' enables msa + boltz + esm + esmfold together so MSA's
             YAMLs feed Boltz/ESM/ESMFold in one shot (Option A calibration).
  gpu_type   One of: a100, h100, h200  (or any partition name to use literally)
  fasta_dir  Directory of .fasta files (real proteins, varied length)
  output_dir Optional output root. Defaults to /tmp/protforge_calib_<gpu>_<ts>
             — override with a SHARED filesystem path (compute nodes can't
             read login-node /tmp).

Env vars (all optional):
  SLURM_ACCOUNT          SLURM account.                       Default: kempner_bsabatini_lab
  CALIB_MAX_JOBS         Cap concurrent SLURM jobs.           Default: 4
  PROTFORGE_LOG_DIR      slurm.log_dir override.              Default: /n/home06/\$USER/job_logs
  PROTFORGE_ESM_ENV      ESM conda env path override.         Default: /n/home06/\$USER/envs/esm
  PROTFORGE_ESMFOLD_ENV  ESMFold conda env path override.     Default: shared lab env
                         (/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge/envs/esmfold)
EOF
    exit 1
}

[[ $# -lt 3 ]] && usage

STAGE="$1"
GPU_TYPE="$2"
FASTA_DIR="$(realpath "$3")"
OUT_ROOT="${4:-/tmp/protforge_calib_${GPU_TYPE}_$(date +%Y%m%d_%H%M%S)}"
MAX_JOBS="${CALIB_MAX_JOBS:-10}"

case "$STAGE" in
    msa|boltz|esm|esmfold|es|all) ;;
    *) echo "Unknown stage: $STAGE"; usage ;;
esac

[[ -d "$FASTA_DIR" ]] || { echo "fasta_dir does not exist: $FASTA_DIR"; exit 1; }

# Resolve which pipeline toggles to flip on. 'all' covers everything except
# 'es' (CPU-only and rarely the bottleneck for a GPU calibration).
_pipe_flag() {
    local s="$1"
    if [[ "$STAGE" == "all" ]]; then
        [[ "$s" != "es" ]] && echo true || echo false
    else
        [[ "$STAGE" == "$s" ]] && echo true || echo false
    fi
}

# Map gpu_type to partition (matches webapp/scaling_models.yaml gpu_specs).
case "$GPU_TYPE" in
    a100)        PARTITION="kempner" ;;
    a100_80)     PARTITION="kempner" ;;
    a100_mig)    PARTITION="kempner_requeue" ;;
    h100)        PARTITION="kempner_h100" ;;
    h200)        PARTITION="kempner_h200" ;;   # confirm against `sinfo`
    *)           PARTITION="$GPU_TYPE" ;;       # treat as literal partition name
esac

# Account default is the kempner-prefixed name because Kempner GPU partitions
# (kempner, kempner_h100, kempner_requeue) require the kempner-prefixed
# account, NOT the sacctmgr default which is the lab's plain account.
# Override with: SLURM_ACCOUNT=<your_kempner_account> bash slurm_scripts/calibrate.sh ...
ACCOUNT="${SLURM_ACCOUNT:-kempner_bsabatini_lab}"

# Shared Kempner paths from config.template.yaml — stable across users.
KEMPNER_MMSEQ2_DB="/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db"
KEMPNER_COLABFOLD_DB="/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/colabfold_db"
KEMPNER_COLABFOLD_BIN="/n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz/localcolabfold/colabfold-conda/bin"
KEMPNER_BOLTZ_CACHE="/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db"
KEMPNER_BOLTZ_ENV="/n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz"
KEMPNER_ESM_CACHE="/n/holylfs06/LABS/bsabatini_lab/Everyone/esm_models_cache"

# User-specific paths (env overrides take precedence).
SLURM_LOG_DIR="${PROTFORGE_LOG_DIR:-/n/home06/$USER/job_logs}"
ESM_ENV="${PROTFORGE_ESM_ENV:-/n/home06/$USER/envs/esm}"
# ESMFold default points at the shared lab env that the webapp uses
# (webapp/app.py:1383). The user-specific path in config.template.yaml
# is misleading — most users do not have their own ESMFold env.
ESMFOLD_ENV="${PROTFORGE_ESMFOLD_ENV:-/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge/envs/esmfold}"

mkdir -p "$OUT_ROOT"
CALIB_CFG="$OUT_ROOT/config.yaml"

# Per-stage partition pin. 'all' pins every GPU stage to the chosen partition;
# a single-stage run pins only that stage (keeps unused stages off the GPU
# queue if anyone enables them later via config edit).
if [[ "$STAGE" == "all" ]]; then
    PER_STAGE_PARTITIONS="$(cat <<EOF
  msa:
    partition: $PARTITION
  boltz:
    partition: $PARTITION
  esm:
    partition: $PARTITION
  esmfold:
    partition: $PARTITION
EOF
)"
else
    PER_STAGE_PARTITIONS="  $STAGE:
    partition: $PARTITION"
fi

cat > "$CALIB_CFG" <<EOF
# Auto-generated by slurm_scripts/calibrate.sh — do not hand-edit.
# Fully self-contained: shared Kempner paths baked in from config.template.yaml.

pipeline:
  msa: $(_pipe_flag msa)
  boltz: $(_pipe_flag boltz)
  esm: $(_pipe_flag esm)
  esmfold: $(_pipe_flag esmfold)
  es: $(_pipe_flag es)

input:
  fasta_dir: $FASTA_DIR

output:
  parent_dir: $OUT_ROOT/run

slurm:
  partition: $PARTITION
  account: $ACCOUNT
  log_dir: $SLURM_LOG_DIR
$PER_STAGE_PARTITIONS
  # Calibration-generous resource ceilings — production rule defaults
  # (Boltz 16 GB / 60 min) are tuned for 25 sequences per chunk where each
  # sequence gets a smaller share. Calibration runs one seq per chunk on
  # potentially long proteins, so request a full H100 and 4 hours to avoid
  # OOM/TIMEOUT. Snakemake's benchmark: directive still captures actual
  # usage; SLURM only allocates what we ask, so over-asking is fine.
  resources:
    boltz:
      mem_mb: 80000
      runtime: 240
      cpus_per_task: 8
      gpus: 1
    esmfold:
      mem_mb: 80000
      runtime: 120
      cpus_per_task: 8
      gpus: 1

# Calibration sweep settings — one sequence per chunk gives one benchmark
# TSV row per length value, which is exactly what we want for fitting.
msa:
  max_files_per_job: 1
  mmseq2_db: $KEMPNER_MMSEQ2_DB
  colabfold_db: $KEMPNER_COLABFOLD_DB
  colabfold_bin: $KEMPNER_COLABFOLD_BIN

boltz:
  max_files_per_job: 1
  recycling_steps: 10
  diffusion_samples: 25
  num_runs: 1
  cache_dir: $KEMPNER_BOLTZ_CACHE
  colabfold_db: $KEMPNER_COLABFOLD_DB
  env_path: $KEMPNER_BOLTZ_ENV

esm:
  num_chunks: 100              # capped at file count by chunker → 1 seq/chunk
  env_path: $ESM_ENV
  cache_dir: $KEMPNER_ESM_CACHE

esmfold:
  num_chunks: 100
  input_type: yaml             # picks up sequences/<seq>/<seq>.yaml from MSA
  env_path: $ESMFOLD_ENV
  cache_dir: $KEMPNER_ESM_CACHE
EOF

echo ">>> Calibration config: $CALIB_CFG"
echo ">>> Stage:              $STAGE"
echo ">>> Partition:          $PARTITION"
echo ">>> Account:            $ACCOUNT"
echo ">>> Concurrency cap:    $MAX_JOBS"
echo ">>> FASTA dir:          $FASTA_DIR"
echo ">>> Output root:        $OUT_ROOT"
echo

# Run the pipeline. --jobs overrides the profile's 100-job cap so calibration
# stays a polite cluster citizen (default 4; bump via CALIB_MAX_JOBS=N).
snakemake --profile profiles/slurm/ \
    --configfile "$CALIB_CFG" \
    --jobs "$MAX_JOBS" \
    --rerun-triggers mtime

# Summarize benchmarks across every stage that ran.
SUMMARY="$OUT_ROOT/summary.txt"
if [[ "$STAGE" == "all" ]]; then
    SUMMARY_STAGES="msa boltz esm esmfold"
else
    SUMMARY_STAGES="$STAGE"
fi
{
    echo "ProtForge calibration — $STAGE on $GPU_TYPE ($PARTITION)"
    echo "==============================================="
    echo "Output root: $OUT_ROOT"
    echo "Started:     $(date)"
    echo
    for s in $SUMMARY_STAGES; do
        BENCH_DIR="$OUT_ROOT/run/benchmarks/$s"
        if [[ -d "$BENCH_DIR" ]]; then
            echo "[$s] Per-rule benchmark TSVs ($BENCH_DIR):"
            for tsv in "$BENCH_DIR"/*.tsv; do
                [[ -f "$tsv" ]] || continue
                echo "  $tsv"
                head -2 "$tsv" | column -t -s $'\t'
            done
            echo
        else
            echo "[$s] No benchmarks/ found at $BENCH_DIR — did this stage run?"
        fi
    done
} > "$SUMMARY"

echo
echo ">>> Done. Summary: $SUMMARY"
