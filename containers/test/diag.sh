#!/usr/bin/env bash
# Diagnostic dump: probe the rebuilt SIF + grab the rule log from a run dir.
# Writes one file with everything. Run on a login node — no GPU needed for
# any of these checks.
#
# Usage:
#   bash containers/test/diag.sh                     # uses most recent run dir
#   bash containers/test/diag.sh -r <RUN_DIR>        # explicit run dir
#   bash containers/test/diag.sh -r <RUN_DIR> -o /path/to/out.txt
#
# Pre-rebuild gotcha to confirm: does the SIF's `mmseqs` list `gpuserver`?
# The GPU build does; the CPU/avx2 build doesn't. That's the single-command
# test for "did the rebuild pick up the def-file swap".

set -euo pipefail

USER_LAB_ROOT="/n/holylfs06/LABS/bsabatini_lab/Everyone/${USER}"
SIF="${SIF:-${USER_LAB_ROOT}/ProtForge/sifs/protforge-gpu.sif}"
RUN_DIR=""
OUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--run-dir) RUN_DIR="$2"; shift 2 ;;
        -o|--out)     OUT="$2"; shift 2 ;;
        -h|--help)    sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Pick most recent run dir if not specified.
if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=$(ls -1dt "${USER_LAB_ROOT}/protforge-baseline"/20* 2>/dev/null | head -1 || true)
fi
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
    echo "ERROR: no run dir found. Pass -r <path>." >&2
    exit 2
fi

# Default output: into the run dir so it travels with the rest of the logs.
if [[ -z "$OUT" ]]; then
    OUT="${RUN_DIR}/diag-$(date +%Y%m%d_%H%M%S).txt"
fi
mkdir -p "$(dirname "$OUT")"

RULE_LOG="${RUN_DIR}/output/logs/msa/colabfold_search_0.log"

{
    echo "=== diag.sh $(date -u +%FT%TZ) ==="
    echo "SIF:     $SIF"
    echo "RUN_DIR: $RUN_DIR"
    echo

    echo "=== 1. mmseqs version (just a sanity-check the binary runs) ==="
    singularity exec "$SIF" mmseqs version || echo "FAILED"
    echo

    echo "=== 2. Does the SIF mmseqs include the GPU subcommand 'gpuserver'? ==="
    echo "    (GPU build: lists 'gpuserver'. CPU/avx2 build: does NOT.)"
    singularity exec "$SIF" mmseqs 2>&1 | grep -iE 'gpu|cuda' || echo "NO MATCHES — this is the CPU/avx2 build"
    echo

    echo "=== 3. Binary strings: CUDA refs ==="
    singularity exec "$SIF" sh -c 'strings $(which mmseqs)' 2>&1 | grep -iE 'libcudart|libcublas|CUDA support' | sort -u | head -10 || true
    echo

    echo "=== 4. SIF build manifest (top 40 lines) ==="
    singularity exec "$SIF" cat /opt/protforge/container-build-manifest.txt 2>&1 | head -40 || echo "no manifest"
    echo

    echo "=== 5. SIF labels (singularity inspect) ==="
    singularity inspect "$SIF" 2>&1 || true
    echo

    echo "=== 6. Rule's own colabfold log (the missing piece) ==="
    if [[ -f "$RULE_LOG" ]]; then
        echo "--- $RULE_LOG ---"
        cat "$RULE_LOG"
    else
        echo "MISSING: $RULE_LOG"
        echo "Looking for alternates..."
        find "${RUN_DIR}/output/logs" -type f -name '*.log' 2>/dev/null | head -10
    fi
} > "$OUT" 2>&1

echo "Wrote $OUT"
echo "Tail:"
tail -5 "$OUT"
