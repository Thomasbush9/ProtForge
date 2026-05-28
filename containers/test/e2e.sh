#!/usr/bin/env bash
# Stage-1 E2E baseline test for protforge-gpu.sif.
#
# Runs a small FASTA set through MSA -> Boltz -> ESM via the SIF + Snakemake +
# SLURM, end-to-end. ESMFold and ES are intentionally disabled for the first
# baseline (ESMFold uses the same HF cache as ESM, so a working ESM run is
# evidence that the cache mount and offline flags are correct; layer ESMFold
# in as a follow-up). Closes Brief #8.
#
# What this script does:
#   1. Validates the SIF + bind mounts exist.
#   2. Generates a session config under $TEST_ROOT/<timestamp>/config.yaml
#      with absolute paths (no $HOME bind, by request).
#   3. Captures `pip freeze` from inside the SIF -> requirements-container.lock
#      (seed for PR-2's lockfile).
#   4. Runs `snakemake -n` to validate the DAG.
#   5. Without --launch: prints the launch command and stops.
#      With --launch:    runs snakemake for real and tees output to the log.
#
# Logging:
#   ALL stdout + stderr from the moment the run dir is created go to
#   ${RUN_DIR}/e2e.log via tee. Includes setup, dry-run, and the real
#   snakemake invocation if --launch is passed. Per-rule SLURM logs land
#   separately in ${RUN_DIR}/job_logs/ (the snakemake-slurm profile owns
#   that). Errors before the run dir exists (missing SIF, missing input)
#   stay on the terminal only — they're instantaneous and self-evident.
#
# Usage:
#   bash containers/test/e2e.sh -i /path/to/10_fastas/            # setup + dry-run
#   bash containers/test/e2e.sh -i /path/to/10_fastas/ --launch   # setup + dry-run + real run
#
# Optional overrides (env vars):
#   SIF        Path to the SIF (default: $TEST_ROOT_PARENT/ProtForge/sifs/protforge-gpu.sif)
#   TEST_ROOT  Where outputs land (default: /n/holylfs06/LABS/bsabatini_lab/Everyone/$USER/protforge-baseline)
#
# Run from the repo root. The host venv (requirements-host.txt) must be active
# so `snakemake` is on PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_DIR=""
LAUNCH=0
TEST_ROOT_PARENT="/n/holylfs06/LABS/bsabatini_lab/Everyone/${USER}"
TEST_ROOT="${TEST_ROOT:-${TEST_ROOT_PARENT}/protforge-baseline}"
SIF="${SIF:-${TEST_ROOT_PARENT}/ProtForge/sifs/protforge-gpu.sif}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input) INPUT_DIR="$2"; shift 2 ;;
        --launch)   LAUNCH=1; shift ;;
        -h|--help)  sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Pre-RUN_DIR validation: errors here stay on terminal only.
if [[ -z "$INPUT_DIR" ]]; then
    echo "ERROR: -i <input_fasta_dir> is required" >&2
    exit 2
fi
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: input dir does not exist: $INPUT_DIR" >&2
    exit 2
fi
if [[ ! -f "$SIF" ]]; then
    echo "ERROR: SIF not found at $SIF" >&2
    echo "Set SIF=<path> to override." >&2
    exit 2
fi

# Bind targets — sourced from config.yaml, paths confirmed 2026-05-28.
# Two different 'Everyone' partitions: kempner_shared for MSA+Boltz, bsabatini
# for ESM/ESMFold cache.
MMSEQS2_DB="/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db"
COLABFOLD_DB="/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/colabfold_db"
BOLTZ_DB="/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db"
COLABFOLD_BIN="/n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz/localcolabfold/colabfold-conda/bin"
ESM_CACHE="/n/holylfs06/LABS/bsabatini_lab/Everyone/esm_models_cache"
USER_LAB_ROOT="/n/holylfs06/LABS/bsabatini_lab/Everyone/${USER}"

for p in "$MMSEQS2_DB" "$COLABFOLD_DB" "$BOLTZ_DB" "$ESM_CACHE" "$USER_LAB_ROOT"; do
    if [[ ! -d "$p" ]]; then
        echo "ERROR: required path missing: $p" >&2
        exit 2
    fi
done

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${TEST_ROOT}/${TS}"
OUTPUT_DIR="${RUN_DIR}/output"
LOG_DIR="${RUN_DIR}/job_logs"
CONFIG="${RUN_DIR}/config.yaml"
LOCKFILE="${RUN_DIR}/requirements-container.lock"
E2E_LOG="${RUN_DIR}/e2e.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Start logging from here on. `tee -a` keeps the log if --launch is re-run
# against an existing RUN_DIR (won't happen with default TS, but defensive).
# `exec` rewires the script's own fds, so every subsequent command (including
# subshells and the eventual snakemake call) flows through tee. Stderr is
# folded into stdout via 2>&1 BEFORE the redirect so order is preserved.
exec > >(tee -a "$E2E_LOG") 2>&1

log_section() { printf '\n=== [%s] %s ===\n' "$(date +%H:%M:%S)" "$*"; }

log_section "Baseline E2E setup"
echo "Run dir:  ${RUN_DIR}"
echo "Input:    ${INPUT_DIR}"
echo "Output:   ${OUTPUT_DIR}"
echo "SIF:      ${SIF}"
echo "Config:   ${CONFIG}"
echo "Log:      ${E2E_LOG}"
echo "Launch:   $([[ $LAUNCH == 1 ]] && echo 'yes' || echo 'no (dry-run only)')"

# Bind paths — host:host throughout so config paths work unchanged inside the
# container. No /n/home06 by design (per 2026-05-28 decision).
BIND_PATHS="${MMSEQS2_DB}:ro,${COLABFOLD_DB}:ro,${BOLTZ_DB}:ro,${ESM_CACHE}:ro,${USER_LAB_ROOT}"

cat > "$CONFIG" <<EOF
# Stage-1 E2E baseline config. Generated by containers/test/e2e.sh on ${TS}.
# Pipeline: MSA + Boltz + ESM through protforge-gpu.sif.
# ESMFold + ES disabled for the baseline; layer in once ESM is green.

pipeline:
  msa: true
  boltz: true
  esm: true
  esmfold: false
  es: false

input:
  fasta_dir: ${INPUT_DIR}

output:
  parent_dir: ${OUTPUT_DIR}

msa:
  max_files_per_job: 5
  array_max_concurrency: 5
  mmseq2_db: ${MMSEQS2_DB}
  colabfold_db: ${COLABFOLD_DB}
  colabfold_bin: ${COLABFOLD_BIN}

boltz:
  max_files_per_job: 5
  array_max_concurrency: 5
  recycling_steps: 3
  diffusion_samples: 5
  num_runs: 1
  cache_dir: ${BOLTZ_DB}
  colabfold_db: ${COLABFOLD_DB}

esm:
  num_chunks: 1
  array_max_concurrency: 5
  cache_dir: ${ESM_CACHE}

containers:
  gpu: ${SIF}
  runtime: auto
  bind_paths: "${BIND_PATHS}"

slurm:
  # NOTE: the snakemake-executor-plugin-slurm does NOT honor a 'log_dir' key
  # here. Slurm logs are placed via the --slurm-logdir CLI flag (wired below).
  partition: kempner_requeue
  account: kempner_bsabatini_lab
  boltz:
    partition: kempner_h100
EOF

log_section "Capturing pip freeze from the SIF"
# No --nv needed for pip freeze (no GPU op). --cleanenv matches the production
# container_cmd() invocation pattern so the lockfile reflects what rules see.
singularity exec --cleanenv "$SIF" pip freeze > "$LOCKFILE"
echo "Wrote $(wc -l < "$LOCKFILE") packages to ${LOCKFILE}"

cd "$REPO_ROOT"

log_section "Dry run (validates DAG + paths)"
snakemake --profile profiles/slurm/ --configfile "$CONFIG" --slurm-logdir "$LOG_DIR" -n

if [[ $LAUNCH != 1 ]]; then
    log_section "Dry run passed; --launch not set"
    cat <<NEXT
To launch the real run with logging:
  cd ${REPO_ROOT}
  bash containers/test/e2e.sh -i ${INPUT_DIR} --launch

Or invoke snakemake directly (output WILL be appended to ${E2E_LOG}):
  cd ${REPO_ROOT}
  snakemake --profile profiles/slurm/ --configfile ${CONFIG} --slurm-logdir ${LOG_DIR} 2>&1 | tee -a ${E2E_LOG}

Monitor:
  squeue -u \$USER
  tail -F ${LOG_DIR}/*.log                    # SLURM worker stdout (--slurm-logdir)
  tail -F ${OUTPUT_DIR}/logs/*/*.log          # per-rule colabfold/boltz/esm stderr

When the run finishes, write up:
  ~/Documents/Vault/Notes/Lab/protforge/log/${TS%_*}-stage1-e2e-baseline.md
NEXT
    exit 0
fi

log_section "Launching real snakemake run"
# `set +e` so we capture exit code and run on-error reporting even if snakemake
# fails. The trap is belt-and-braces in case of SIGINT during a long run.
trap 'log_section "Interrupted (SIGINT); see ${E2E_LOG} and ${LOG_DIR}"; exit 130' INT
set +e
# --restart-times 0: this is a baseline/debug E2E. Retries truncate the
# rule's {log} on each attempt (`exec > {log}`), so the first-attempt error
# gets overwritten and we end up debugging a stale-state attempt-3 trace.
# Disabling retries here gives a clean single-attempt failure. Profile
# default (`restart-times: 2`) still applies to production runs.
snakemake --profile profiles/slurm/ --configfile "$CONFIG" --slurm-logdir "$LOG_DIR" --restart-times 0
RC=$?
set -e
trap - INT

log_section "Real run finished — exit code ${RC}"

# Centralise every log we produced into ${RUN_DIR}/all_logs/ so there's ONE
# directory to inspect or ship for debugging. Three sources:
#   1. Rule {log} outputs  (output/logs/<stage>/*.log)  — colabfold/boltz/esm traces
#   2. SLURM worker logs   (job_logs/**/*.log)          — snakemake's per-job wrapper
#   3. e2e driver log      (e2e.log)                    — already in ${RUN_DIR}
# Copies (not symlinks) so the dir is self-contained and rsyncable.
ALL_LOGS="${RUN_DIR}/all_logs"
mkdir -p "$ALL_LOGS"
log_section "Centralising logs into ${ALL_LOGS}"
if [[ -d "${OUTPUT_DIR}/logs" ]]; then
    while IFS= read -r -d '' f; do
        rel="${f#${OUTPUT_DIR}/logs/}"
        flat="rule_${rel//\//_}"
        cp "$f" "${ALL_LOGS}/${flat}"
    done < <(find "${OUTPUT_DIR}/logs" -type f -name '*.log' -print0 2>/dev/null)
fi
if [[ -d "$LOG_DIR" ]]; then
    while IFS= read -r -d '' f; do
        rel="${f#${LOG_DIR}/}"
        flat="slurm_${rel//\//_}"
        cp "$f" "${ALL_LOGS}/${flat}"
    done < <(find "$LOG_DIR" -type f -name '*.log' -print0 2>/dev/null)
fi
cp "$E2E_LOG" "${ALL_LOGS}/e2e.log"
echo "Collected $(find "$ALL_LOGS" -maxdepth 1 -type f | wc -l | tr -d ' ') file(s):"
ls -1 "$ALL_LOGS"

if [[ $RC -eq 0 ]]; then
    echo ""
    echo "SUCCESS. Outputs at ${OUTPUT_DIR}"
    if [[ -f "${OUTPUT_DIR}/benchmark_summary.txt" ]]; then
        echo ""
        echo "--- benchmark_summary.txt ---"
        cat "${OUTPUT_DIR}/benchmark_summary.txt"
    fi
else
    echo ""
    echo "FAILED. Everything you need is in ONE place:"
    echo "  ${ALL_LOGS}/"
    echo ""
    echo "Most likely the real error is in a 'rule_msa_*' or 'rule_boltz_*' file there."
fi

echo ""
echo "Write-up target:"
echo "  ~/Documents/Vault/Notes/Lab/protforge/log/${TS%_*}-stage1-e2e-baseline.md"

exit $RC
