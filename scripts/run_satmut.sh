#!/usr/bin/env bash
# Standalone saturation-mutagenesis logits job.
#
# Runs the ESM-C container with --save-logits on a one-sequence input dir,
# producing logits.npy (+ aa_token_ids.json). The Len x 20-AA LLR matrix is
# derived afterwards by the webapp (cheap, pure NumPy on the login node), so
# this job only needs the GPU forward pass.
#
# No #SBATCH headers — the caller (webapp/satmut.py) passes sbatch flags from
# the session config (partition/account/gpus/mem/time). Mirrors the singularity
# invocation in workflow/rules/esmc.smk.

set -euo pipefail

usage() {
  echo "usage: $0 --sif S --cache C --input DIR --output DIR --runner R \\" >&2
  echo "          [--size 6B|600M|300M] [--runtime singularity|apptainer]" >&2
  exit 1
}

SIF="" CACHE="" INPUT="" OUTPUT="" RUNNER="" SIZE="6B" RUNTIME="singularity"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sif) SIF="$2"; shift 2;;
    --cache) CACHE="$2"; shift 2;;
    --input) INPUT="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    --runner) RUNNER="$2"; shift 2;;
    --size) SIZE="$2"; shift 2;;
    --runtime) RUNTIME="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; usage;;
  esac
done
[[ -n "$SIF" && -n "$CACHE" && -n "$INPUT" && -n "$OUTPUT" && -n "$RUNNER" ]] || usage

mkdir -p "$OUTPUT"
export CUDA_VISIBLE_DEVICES=0

"$RUNTIME" exec --nv --cleanenv \
  --env HF_HOME=/models/hf \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  -B "$CACHE:/models/hf:ro" \
  -B "$INPUT:$INPUT:ro" \
  -B "$OUTPUT:$OUTPUT" \
  -B "$RUNNER:/opt/run_batch_esmc.py:ro" \
  "$SIF" \
  python /opt/run_batch_esmc.py \
    --cache /models/hf \
    --input-dir "$INPUT" \
    --output-dir "$OUTPUT" \
    --save-logits \
    --size "$SIZE"

echo "satmut logits job done: $OUTPUT"
