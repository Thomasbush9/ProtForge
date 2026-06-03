#!/usr/bin/env bash
# run_stage.sh — driver for the stair benchmark.
#
# 1. (once) builds the staircase: pick_rungs.py + chunk_fastas.py.
# 2. submits one --exclusive SLURM array per stage (one task per rung).
#    Model stages (boltz/esmfold/esmc_*) depend on the MSA array (afterok),
#    because they consume the YAML the MSA stage mints.
#
# SLURM partition/account/log_dir + per-stage runtime are read from --config
# (slurm.* and slurm.resources.<stage>.runtime) and passed as sbatch flags,
# overriding the placeholders in stair_bench.slrm.
#
# Usage:
#   run_stage.sh --run-dir DIR --config config.yaml \
#       [--input-dir /path/ga_fasta] [--gpu-type h100] \
#       [--stages "msa boltz esmfold esmc_300M esmc_600M esmc_6B"] \
#       [--min 200 --max 3000 --step 200] [--setup-only] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BENCH_ONE="$SCRIPT_DIR/bench_one.sh"
SLRM="$SCRIPT_DIR/stair_bench.slrm"

RUN_DIR="" CONFIG="" INPUT_DIR="" GPU_TYPE="h100"
STAGES="msa boltz esmfold esmc_300M esmc_600M esmc_6B"
MIN=200 MAX=3000 STEP=200 SETUP_ONLY=0 DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)    RUN_DIR="$2"; shift 2 ;;
    --config)     CONFIG="$2"; shift 2 ;;
    --input-dir)  INPUT_DIR="$2"; shift 2 ;;
    --gpu-type)   GPU_TYPE="$2"; shift 2 ;;
    --stages)     STAGES="$2"; shift 2 ;;
    --min)        MIN="$2"; shift 2 ;;
    --max)        MAX="$2"; shift 2 ;;
    --step)       STEP="$2"; shift 2 ;;
    --setup-only) SETUP_ONLY=1; shift ;;
    --dry-run)    DRY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_DIR" && -n "$CONFIG" ]] || { echo "need --run-dir and --config" >&2; exit 2; }

cfg() {  # cfg KEY [DEFAULT]
  if [[ $# -ge 2 ]]; then python3 "$SCRIPT_DIR/cfg_get.py" "$CONFIG" "$1" "$2"
  else python3 "$SCRIPT_DIR/cfg_get.py" "$CONFIG" "$1"; fi
}

STAIR="$RUN_DIR/staircase"
CHUNKS="$RUN_DIR/chunks"

# ---- 1. setup (host-side, idempotent) ---------------------------------------
if [[ ! -f "$STAIR/rungs.csv" ]]; then
  [[ -n "$INPUT_DIR" ]] || { echo "no staircase yet — pass --input-dir to build it" >&2; exit 2; }
  echo "[setup] picking rungs from $INPUT_DIR ($MIN..$MAX step $STEP)"
  python3 "$SCRIPT_DIR/pick_rungs.py" --input_dir "$INPUT_DIR" \
    --output_dir "$STAIR" --min "$MIN" --max "$MAX" --step "$STEP"
else
  echo "[setup] staircase exists: $STAIR/rungs.csv (reusing)"
fi

if [[ ! -f "$CHUNKS/manifest.txt" ]]; then
  echo "[setup] chunking staircase (one file per chunk)"
  python3 "$REPO_ROOT/workflow/scripts/chunk_fastas.py" \
    --input_dir "$STAIR" --output_dir "$CHUNKS" --max_files_per_job 1
else
  echo "[setup] chunks exist: $CHUNKS (reusing)"
fi

NRUNGS=$(find "$STAIR" -maxdepth 1 -name 'rung_*.fasta' | wc -l | tr -d ' ')
[[ "$NRUNGS" -gt 0 ]] || { echo "no rung_*.fasta in $STAIR" >&2; exit 1; }
echo "[setup] $NRUNGS rungs ready"
[[ "$SETUP_ONLY" == 1 ]] && { echo "[setup] --setup-only: done"; exit 0; }

# ---- 2. submit per-stage arrays ---------------------------------------------
PART="$(cfg slurm.partition)"
ACCT="$(cfg slurm.account)"
LOGDIR="$(cfg slurm.log_dir .)"
mkdir -p "$LOGDIR" "$RUN_DIR/outputs"

submit_stage() {  # submit_stage STAGE [DEP_JOBID]
  local stage="$1" dep="${2:-}"
  local runtime_min; runtime_min="$(cfg "slurm.resources.${stage}.runtime" 120)"
  local cpus;        cpus="$(cfg "slurm.resources.${stage}.cpus_per_task" 8)"
  local cmd=(sbatch --parsable
    --job-name="stair_${stage}_${GPU_TYPE}"
    --partition="$PART" --account="$ACCT"
    --array="0-$((NRUNGS-1))"
    --time="$runtime_min" --cpus-per-task="$cpus"
    --output="$LOGDIR/stair_${stage}_%A_%a.out"
    --export="ALL,STAGE=${stage},RUN_DIR=${RUN_DIR},CONFIG=${CONFIG},GPU_TYPE=${GPU_TYPE},BENCH_ONE=${BENCH_ONE}")
  [[ -n "$dep" ]] && cmd+=(--dependency="afterok:${dep}")
  cmd+=("$SLRM")
  if [[ "$DRY" == 1 ]]; then
    # command -> stderr so the captured stdout is ONLY the (fake) job id,
    # mirroring sbatch --parsable in real mode.
    { printf '[dry-run] %q ' "${cmd[@]}"; echo; } >&2
    echo "DRYJOB_${stage}"; return
  fi
  "${cmd[@]}"
}

MSA_JID=""
# Submit MSA first if requested; capture its jobid for downstream deps.
if [[ " $STAGES " == *" msa "* ]]; then
  MSA_JID="$(submit_stage msa)"
  echo "[submit] msa array -> job $MSA_JID"
fi

for stage in $STAGES; do
  [[ "$stage" == msa ]] && continue
  jid="$(submit_stage "$stage" "$MSA_JID")"
  echo "[submit] $stage array -> job $jid${MSA_JID:+ (afterok:$MSA_JID)}"
done

echo "[done] results will accumulate in $RUN_DIR/results.csv"
