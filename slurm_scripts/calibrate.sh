#!/usr/bin/env bash
# calibrate.sh — measure SLURM consumption for one ProtForge stage on one GPU.
#
# Usage:
#   bash slurm_scripts/calibrate.sh <stage> <gpu_type> <fasta_dir> [output_dir]
#
# Example:
#   bash slurm_scripts/calibrate.sh boltz h100 tests/calibration_inputs/fastas
#
# What it does:
#   1. Generates a temp Snakemake config at $CALIB_DIR/config.yaml that:
#        - enables only <stage>
#        - points input.fasta_dir at <fasta_dir>
#        - routes <stage> to the partition matching <gpu_type>
#   2. Runs `snakemake --profile profiles/slurm/` against that config
#   3. After completion, writes a one-pager at $CALIB_DIR/summary.txt with
#      the per-rule walltimes and max RSS values.
#
# After you run this for each (stage × gpu_type) combination you care about,
# point the webapp at $CALIB_DIR via the "Recalibrate from cluster benchmarks"
# button to refit webapp/scaling_models.calibrated.yaml.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 <stage> <gpu_type> <fasta_dir> [output_dir]

  stage     One of: msa, boltz, esm, esmfold, es, all
            'all' enables msa + boltz + esm + esmfold together so MSA's
            YAMLs feed Boltz/ESM/ESMFold in one shot (Option A calibration).
  gpu_type  One of: a100, h100, h200  (or any partition name to use literally)
  fasta_dir Path to a directory of .fasta files (real proteins, varied length)
  output_dir Optional output root. Defaults to /tmp/protforge_calib_<gpu_type>_<timestamp>

Env vars:
  CALIB_MAX_JOBS  Cap concurrent SLURM jobs (default: 4). Overrides the
                  profile's jobs:100 setting so calibration does not crowd
                  the queue. Bump if you have headroom.
EOF
    exit 1
}

[[ $# -lt 3 ]] && usage

STAGE="$1"
GPU_TYPE="$2"
FASTA_DIR="$(realpath "$3")"
OUT_ROOT="${4:-/tmp/protforge_calib_${GPU_TYPE}_$(date +%Y%m%d_%H%M%S)}"
MAX_JOBS="${CALIB_MAX_JOBS:-4}"

case "$STAGE" in
    msa|boltz|esm|esmfold|es|all) ;;
    *) echo "Unknown stage: $STAGE"; usage ;;
esac

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

[[ -d "$FASTA_DIR" ]] || { echo "fasta_dir does not exist: $FASTA_DIR"; exit 1; }

# Map gpu_type to partition (matches webapp/scaling_models.yaml gpu_specs).
case "$GPU_TYPE" in
    a100)        PARTITION="kempner" ;;
    a100_80)     PARTITION="kempner" ;;
    a100_mig)    PARTITION="kempner_requeue" ;;
    h100)        PARTITION="kempner_h100" ;;
    h200)        PARTITION="kempner_h200" ;;   # confirm against `sinfo`
    *)           PARTITION="$GPU_TYPE" ;;       # treat as literal partition name
esac

mkdir -p "$OUT_ROOT"
CALIB_CFG="$OUT_ROOT/config.yaml"

ACCOUNT="${SLURM_ACCOUNT:-${SLURM_DEFAULT_ACCOUNT:-$(sacctmgr -nP show user $USER format=defaultaccount 2>/dev/null | head -n1)}}"
ACCOUNT="${ACCOUNT:-kempner_bsabatini_lab}"

# Build the calibration config. When STAGE=all, every GPU stage targets the
# chosen partition; Snakemake's DAG sequences them via .msa_complete /
# .boltz_complete sentinels.
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
  log_dir: $OUT_ROOT/logs
$PER_STAGE_PARTITIONS

# Stage-specific basics — leave file_per_job small so the calibration sweep
# produces one benchmark TSV row per sequence (clean per-length data).
msa:
  max_files_per_job: 1
boltz:
  max_files_per_job: 1
  recycling_steps: 10
  diffusion_samples: 25
  num_runs: 1
esm:
  num_chunks: 100
esmfold:
  num_chunks: 100
EOF

echo ">>> Calibration config: $CALIB_CFG"
echo ">>> Partition: $PARTITION"
echo ">>> FASTA dir:  $FASTA_DIR"
echo

# Ensure the user has at least pasted in the shared-resource paths from their
# real config (cache_dir, env_path, db paths). Pull them from config.yaml if it
# exists at repo root — calibration uses the same shared resources.
if [[ -f config.yaml ]]; then
    echo ">>> Merging cache/env paths from config.yaml ..."
    python3 - <<PY
import yaml
base = yaml.safe_load(open("config.yaml")) or {}
cal  = yaml.safe_load(open("$CALIB_CFG")) or {}
for stage in ("msa", "boltz", "esm", "esmfold", "es"):
    src = base.get(stage, {}) or {}
    dst = cal.setdefault(stage, {})
    for k in ("cache_dir", "env_path", "mmseq2_db", "colabfold_db",
              "colabfold_bin", "pdanalysis_dir"):
        if k in src and k not in dst:
            dst[k] = src[k]
yaml.safe_dump(cal, open("$CALIB_CFG", "w"), sort_keys=False)
PY
fi

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
echo ">>> Feed this output dir to the webapp: 'Recalibrate from cluster benchmarks'"
