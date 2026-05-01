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

  stage     One of: msa, boltz, esm, esmfold, es
  gpu_type  One of: a100, h100, h200  (or any partition name to use literally)
  fasta_dir Path to a directory of .fasta files (real proteins, varied length)
  output_dir Optional output root. Defaults to /tmp/protforge_calib_<gpu_type>_<timestamp>
EOF
    exit 1
}

[[ $# -lt 3 ]] && usage

STAGE="$1"
GPU_TYPE="$2"
FASTA_DIR="$(realpath "$3")"
OUT_ROOT="${4:-/tmp/protforge_calib_${GPU_TYPE}_$(date +%Y%m%d_%H%M%S)}"

case "$STAGE" in
    msa|boltz|esm|esmfold|es) ;;
    *) echo "Unknown stage: $STAGE"; usage ;;
esac

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

# Build a minimal config that enables only the chosen stage.
cat > "$CALIB_CFG" <<EOF
pipeline:
  msa: $([[ "$STAGE" == "msa" ]] && echo true || echo false)
  boltz: $([[ "$STAGE" == "boltz" ]] && echo true || echo false)
  esm: $([[ "$STAGE" == "esm" ]] && echo true || echo false)
  esmfold: $([[ "$STAGE" == "esmfold" ]] && echo true || echo false)
  es: $([[ "$STAGE" == "es" ]] && echo true || echo false)

input:
  fasta_dir: $FASTA_DIR

output:
  parent_dir: $OUT_ROOT/run

slurm:
  partition: $PARTITION
  account: $ACCOUNT
  log_dir: $OUT_ROOT/logs
  $STAGE:
    partition: $PARTITION

# Stage-specific basics — leave file_per_job small so the calibration sweep
# produces multiple per-chunk benchmark TSV rows for a better fit.
msa:
  max_files_per_job: 1
boltz:
  max_files_per_job: 1
  recycling_steps: 10
  diffusion_samples: 25
  num_runs: 1
esm:
  num_chunks: 5
esmfold:
  num_chunks: 5
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

# Run the pipeline
snakemake --profile profiles/slurm/ \
    --configfile "$CALIB_CFG" \
    --rerun-triggers mtime

# Summarize benchmarks
SUMMARY="$OUT_ROOT/summary.txt"
{
    echo "ProtForge calibration — $STAGE on $GPU_TYPE ($PARTITION)"
    echo "==============================================="
    echo "Output root: $OUT_ROOT"
    echo "Started:     $(date)"
    echo
    BENCH_DIR="$OUT_ROOT/run/benchmarks/$STAGE"
    if [[ -d "$BENCH_DIR" ]]; then
        echo "Per-rule benchmark TSVs ($BENCH_DIR):"
        for tsv in "$BENCH_DIR"/*.tsv; do
            [[ -f "$tsv" ]] || continue
            echo "  $tsv"
            head -2 "$tsv" | column -t -s $'\t'
        done
    else
        echo "No benchmarks/ found at $BENCH_DIR — did the pipeline run?"
    fi
} > "$SUMMARY"

echo
echo ">>> Done. Summary: $SUMMARY"
echo ">>> Feed this output dir to the webapp: 'Recalibrate from cluster benchmarks'"
