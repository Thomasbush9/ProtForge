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

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--image) SIF="$2"; shift 2 ;;
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

mkdir -p "$WORK"

run_in_container() {
    singularity exec --nv \
        -B "${SCRIPT_DIR}":"${SCRIPT_DIR}" \
        -B "${WORK}":"${WORK}" \
        "$SIF" "$@"
}

echo "=== [1/5] GPU visibility ==="
run_in_container nvidia-smi

echo
echo "=== [2/5] PyTorch + CUDA ==="
run_in_container python -c "
import torch
print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
assert torch.cuda.is_available(), 'CUDA not visible inside container'
"

echo
echo "=== [3/5] Tools importable ==="
run_in_container bash -c '
set -e
boltz --help >/dev/null 2>&1 && echo "boltz: OK"
mmseqs version >/dev/null 2>&1 && echo "mmseqs: OK"
command -v colabfold_search >/dev/null && echo "colabfold_search: OK" || echo "colabfold_search: MISSING"
python -c "import esm; print(\"esm SDK: OK\")"
python -c "from transformers import EsmForProteinFolding; print(\"transformers ESMFold: OK\")"
'

echo
echo "=== [4/5] Baked weights load ==="
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
echo "=== [5/5] End-to-end ESMFold fold ==="
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
echo "=== ALL SMOKE TESTS PASSED ==="
