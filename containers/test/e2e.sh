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
#   bash containers/test/e2e.sh -i /path/to/10_fastas/                       # setup + dry-run
#   bash containers/test/e2e.sh -i /path/to/10_fastas/ --launch              # setup + dry-run + real run
#   bash containers/test/e2e.sh -i /path/to/10_fastas/ --skip-msa --launch   # pre-rebuild path (audit H0)
#
# --skip-msa: auto-generate msa:empty YAMLs from the FASTAs and run Boltz+ESM
#   only. Use this while the current SIF still has CPU-only mmseqs2; remove
#   once the GPU-mmseqs rebuild lands.
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
SKIP_MSA=0
TEST_ROOT_PARENT="/n/holylfs06/LABS/bsabatini_lab/Everyone/${USER}"
TEST_ROOT="${TEST_ROOT:-${TEST_ROOT_PARENT}/protforge-baseline}"
SIF="${SIF:-${TEST_ROOT_PARENT}/ProtForge/sifs/protforge-gpu.sif}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)  INPUT_DIR="$2"; shift 2 ;;
        --launch)    LAUNCH=1; shift ;;
        --skip-msa)  SKIP_MSA=1; shift ;;
        -h|--help)   sed -n '2,44p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
echo "MSA:      $([[ $SKIP_MSA == 1 ]] && echo 'SKIPPED (auto-generating msa:empty YAMLs)' || echo 'enabled')"

# Bind paths — host:host throughout so config paths work unchanged inside the
# container. No /n/home06 by design (per 2026-05-28 decision).
BIND_PATHS="${MMSEQS2_DB}:ro,${COLABFOLD_DB}:ro,${BOLTZ_DB}:ro,${ESM_CACHE}:ro,${USER_LAB_ROOT}"

# --skip-msa: convert FASTAs in INPUT_DIR into msa:empty YAMLs and switch the
# config to yaml_dir input + pipeline.msa=false. Used pre-rebuild while the
# SIF still ships CPU-only mmseqs2 (audit H0). Boltz + ESM still exercise the
# container chain end-to-end; MSA validation waits for the GPU-mmseqs rebuild.
PIPELINE_MSA="true"
INPUT_BLOCK="  fasta_dir: ${INPUT_DIR}"
if [[ $SKIP_MSA == 1 ]]; then
    YAML_DIR="${RUN_DIR}/yamls"
    mkdir -p "$YAML_DIR"
    log_section "Generating msa:empty YAMLs from FASTAs"
    python3 - "$INPUT_DIR" "$YAML_DIR" <<'PYEOF'
import sys, re
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
n = 0
for fa in sorted(list(src.glob("*.fasta")) + list(src.glob("*.fa"))):
    text = fa.read_text()
    # Take the FIRST record's header + sequence. Baseline test inputs are
    # single-record FASTAs per the diagram's 1 protein -> 1 file convention.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines or not lines[0].startswith(">"):
        print(f"skip: {fa.name} (no FASTA header)", file=sys.stderr); continue
    name = re.sub(r"[^A-Za-z0-9_-]", "_", lines[0][1:].split()[0]) or fa.stem
    seq = "".join(l for l in lines[1:] if not l.startswith(">"))
    if not seq:
        print(f"skip: {fa.name} (empty sequence)", file=sys.stderr); continue
    (dst / f"{name}.yaml").write_text(
        f"version: 1\nsequences:\n  - protein:\n      id: {name}\n"
        f"      sequence: {seq}\n      msa: empty\n"
    )
    n += 1
print(f"Wrote {n} YAML(s) to {dst}")
PYEOF
    PIPELINE_MSA="false"
    INPUT_BLOCK="  yaml_dir: ${YAML_DIR}"
fi

cat > "$CONFIG" <<EOF
# Stage-1 E2E baseline config. Generated by containers/test/e2e.sh on ${TS}.
# Pipeline (this run): msa=${PIPELINE_MSA}, boltz=true, esm=true.
# ESMFold + ES disabled for the baseline; layer in once ESM is green.

pipeline:
  msa: ${PIPELINE_MSA}
  boltz: true
  esm: true
  esmfold: false
  es: false

input:
${INPUT_BLOCK}

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
snakemake --profile profiles/slurm/ --configfile "$CONFIG" --slurm-logdir "$LOG_DIR"
RC=$?
set -e
trap - INT

log_section "Real run finished — exit code ${RC}"
if [[ $RC -eq 0 ]]; then
    echo "SUCCESS. Outputs at ${OUTPUT_DIR}"
    if [[ -f "${OUTPUT_DIR}/benchmark_summary.txt" ]]; then
        echo ""
        echo "--- benchmark_summary.txt ---"
        cat "${OUTPUT_DIR}/benchmark_summary.txt"
    fi
else
    echo "FAILED. Inspect (in this order — rule logs have the actual error):"
    echo "  ${OUTPUT_DIR}/logs/<stage>/*.log    (rule's own stdout/stderr — colabfold/boltz/esm traces)"
    echo "  ${LOG_DIR}/*.log                    (SLURM-side worker logs, via --slurm-logdir)"
    echo "  ${E2E_LOG}                          (this script + snakemake driver)"
fi

echo ""
echo "Write-up target:"
echo "  ~/Documents/Vault/Notes/Lab/protforge/log/${TS%_*}-stage1-e2e-baseline.md"

exit $RC
