#!/usr/bin/env bash
# Smoke test for protforge-gpu.sif.
#
# Validates:
#   1. Container launches and sees the GPU (`nvidia-smi` works).
#   2. PyTorch sees CUDA.
#   3. Baked tools are importable (boltz CLI, mmseqs CLI, esm SDK, transformers).
#   4. Baked model weights load (ESM-C 600M, ESMFold via HF).
#   5. ESMFold can fold a short sequence end-to-end on GPU.
#
# Does NOT test:
#   - MSA pipeline (needs the ~700 GB colabfold DB bind-mounted).
#   - Boltz pipeline (needs the boltz_db bind-mounted).
#
# Usage:
#   bash containers/test/smoke.sh                  # default: PROTFORGE_SIF_DIR, PROTFORGE_ROOT/sifs, or ~/sifs
#   bash containers/test/smoke.sh -i /path/to/sif  # custom path
#
# Run on a GPU node (e.g. salloc -p kempner_h100 --gres=gpu:1 -t 30 --mem=32G).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF=""
WORK="${SCRIPT_DIR}/_smoke_out"
LOG_FILE=""
LOG_DISABLED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--image) SIF="$2"; shift 2 ;;
        --log)      LOG_FILE="$2"; shift 2 ;;
        --no-log)   LOG_DISABLED=1; shift ;;
        -h|--help)  sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$SIF" ]]; then
    if [[ -n "${PROTFORGE_SIF_DIR:-}" ]]; then
        SIF="${PROTFORGE_SIF_DIR%/}/protforge-gpu.sif"
    elif [[ -n "${PROTFORGE_ROOT:-}" ]]; then
        SIF="${PROTFORGE_ROOT%/}/sifs/protforge-gpu.sif"
    else
        SIF="${HOME}/sifs/protforge-gpu.sif"
    fi
fi

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: image not found at $SIF" >&2
    echo "Build first: export PROTFORGE_ROOT=... && bash containers/build.sh  (or bash containers/build.sh -o ...)" >&2
    exit 1
fi

# Auto-log: same pattern as build.sh. Default location is sibling of sifs/
# under PROTFORGE_ROOT (or wherever the SIF lives). Override with --log,
# disable with --no-log.
if (( ! LOG_DISABLED )) && [[ -z "$LOG_FILE" ]]; then
    if [[ -n "${PROTFORGE_ROOT:-}" ]]; then
        log_base="${PROTFORGE_ROOT%/}/smoke-logs"
    else
        # SIF parent's parent: e.g. .../container/sifs/sif -> .../container/
        log_base="$(dirname "$(dirname "$SIF")")/smoke-logs"
    fi
    mkdir -p "$log_base"
    LOG_FILE="${log_base}/smoke-$(date +%Y-%m-%dT%H-%M-%S).log"
fi
if (( ! LOG_DISABLED )); then
    mkdir -p "$(dirname "$LOG_FILE")"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Logging to  : $LOG_FILE"
    echo "             (rerun with --no-log to disable, or --log PATH to override)"
fi

mkdir -p "$WORK"

# Match the production container_cmd flags: --cleanenv (host env stripped),
# --nv (GPU), TMPDIR=/tmp + node-local scratch at /tmp. Mirrors what
# Snakefile:container_cmd() emits so the smoke test exercises the same
# invocation surface as the real rules.
#
# HF_HUB_OFFLINE + TRANSFORMERS_OFFLINE: the baked HF cache under
# /opt/weights/hf is read-only inside the SIF, but HF's from_pretrained
# tries to acquire a write lock under .locks/ before serving from cache
# (OSError: Read-only file system). Offline mode skips the lock and reads
# cached files directly. Override per-call with --env HF_HUB_OFFLINE=0 if
# you bind-mount a writable HF cache.
RUNTIME="${PROTFORGE_RUNTIME:-singularity}"
run_in_container() {
    "$RUNTIME" exec --nv --cleanenv \
        --env TMPDIR=/tmp \
        --env HF_HUB_OFFLINE=1 \
        --env TRANSFORMERS_OFFLINE=1 \
        -B "${SCRIPT_DIR}":"${SCRIPT_DIR}" \
        -B "${WORK}":"${WORK}" \
        -B "${SLURM_TMPDIR:-/tmp}":/tmp \
        "$SIF" "$@"
}

echo "=== [0/6] Runtime + image identity ==="
"$RUNTIME" --version 2>&1 | head -1
"$RUNTIME" inspect "$SIF" | head -20

echo
echo "=== [1/6] GPU visibility ==="
run_in_container nvidia-smi

echo
echo "=== [2/6] PyTorch + CUDA ==="
run_in_container python -c "
import torch
print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
assert torch.cuda.is_available(), 'CUDA not visible inside container'
"

echo
echo "=== [3/6] Tools importable ==="
# Each check is wrapped so failures print loudly and abort the test.
# Previously `cmd && echo OK` swallowed failures because bash set -e
# does not fire on the failing left side of an && chain.
run_in_container bash -c '
fail=0
check() {
    local name="$1"; shift
    if out="$("$@" 2>&1)"; then
        echo "$name: OK"
    else
        echo "$name: FAILED" >&2
        echo "----- output -----" >&2
        echo "$out" >&2
        echo "------------------" >&2
        fail=1
    fi
}
check "boltz"              boltz --help
check "mmseqs"             mmseqs version
check "colabfold_search"   bash -c "command -v colabfold_search >/dev/null"
check "esm SDK"            python -c "import esm"
check "transformers ESMFold" python -c "from transformers import EsmForProteinFolding"
exit $fail
'

echo
echo "=== [4/6] Baked weights load ==="
run_in_container python -c "
import os
print(f'HF_HOME={os.environ.get(\"HF_HOME\")}')
from transformers import AutoTokenizer, EsmForProteinFolding
tok = AutoTokenizer.from_pretrained('facebook/esmfold_v1')
print('ESMFold tokenizer: OK')
from esm.models.esmc import ESMC
m = ESMC.from_pretrained('esmc_600m')
print('ESM-C 600M: OK')
"

echo
echo "=== [5/6] End-to-end ESMFold fold ==="
run_in_container python - <<PYEOF
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

seq = "MKTIIALSYIFCLVFADYKDDDDKMRGSHHHHHHGSDYDIPTTENLYFQ"
tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).cuda().eval()
with torch.no_grad():
    out = model(**tok([seq], return_tensors="pt", add_special_tokens=False).to("cuda"))
plddt = out["plddt"].mean().item()
print(f"Mean pLDDT for smoke seq: {plddt:.3f}")
assert plddt > 0.0, "pLDDT not in valid range"
print("ESMFold end-to-end: OK")
PYEOF

echo
echo "=== [6/6] Host env isolation (--cleanenv regression) ==="
# Regression guard for audit item H1 (vault container-audit.md). If
# --cleanenv is silently dropped, a poisoned host PYTHONPATH would land
# in sys.path inside the container; the assertion below would then fail.
PYTHONPATH=/host/leaky/smoke run_in_container python -c "
import sys, os
leaky = '/host/leaky/smoke'
assert leaky not in sys.path, f'host PYTHONPATH leaked into container: {sys.path}'
assert os.environ.get('PYTHONPATH', '').find(leaky) < 0, f'PYTHONPATH env leaked: {os.environ.get(\"PYTHONPATH\")}'
print('env isolation: OK')
"

echo
echo "=== ALL SMOKE TESTS PASSED ==="
[[ -n "$LOG_FILE" ]] && echo "Smoke log saved: $LOG_FILE"
