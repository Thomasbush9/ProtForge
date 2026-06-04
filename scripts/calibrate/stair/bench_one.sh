#!/usr/bin/env bash
# bench_one.sh — run ONE stage on ONE staircase rung, measured.
#
# Measures, for a single (stage, rung):
#   wall_total_s       process start -> exit around the singularity exec
#   infer_s            model-call-only time (esmfold/esmc emit BENCH_INFER_S;
#                      msa/boltz are black-box CLIs -> infer_s = wall_total_s)
#   vram_peak_mib      peak nvidia-smi memory.used (reserved VRAM, the --gres
#                      sizing number) — sampled across the whole node; correct
#                      under --exclusive where only this job touches the GPUs
#   host_ram_peak_mib  cgroup peak (v2 memory.peak / v1 max_usage_in_bytes)
#   status             ok | oom | oom-host | timeout | error
#
# Image / DB / cache paths come from config.yaml (nothing hardcoded). The
# singularity exec lines mirror containers/test/*_test_image.sh and msa.smk.
# A row is appended to <run-dir>/results.csv under flock (array-write safe).
#
# Usage:
#   bench_one.sh --stage msa|boltz|esmfold|esmc_300M|esmc_600M|esmc_6B \
#                --rung-idx N --run-dir DIR --config config.yaml --gpu-type h100
set -uo pipefail

# Force C locale: under a comma-decimal locale (e.g. it_IT) awk misparses
# "12.5" as 12 and prints "12,0000", corrupting infer_s/wall_s and the CSV.
export LC_ALL=C LANG=C

# ---- args -------------------------------------------------------------------
STAGE="" RUNG_IDX="" RUN_DIR="" CONFIG="" GPU_TYPE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)     STAGE="$2"; shift 2 ;;
    --rung-idx)  RUNG_IDX="$2"; shift 2 ;;
    --run-dir)   RUN_DIR="$2"; shift 2 ;;
    --config)    CONFIG="$2"; shift 2 ;;
    --gpu-type)  GPU_TYPE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
for v in STAGE RUNG_IDX RUN_DIR CONFIG GPU_TYPE; do
  [[ -n "${!v}" ]] || { echo "missing --${v,,}" >&2; exit 2; }
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONTAINERS="$REPO_ROOT/containers"
WF_SCRIPTS="$REPO_ROOT/workflow/scripts"

cfg() {  # cfg KEY [DEFAULT]
  if [[ $# -ge 2 ]]; then python3 "$SCRIPT_DIR/cfg_get.py" "$CONFIG" "$1" "$2"
  else python3 "$SCRIPT_DIR/cfg_get.py" "$CONFIG" "$1"; fi
}

RUNTIME="$(cfg containers.runtime auto)"
if [[ "$RUNTIME" == auto ]]; then
  if command -v singularity >/dev/null 2>&1; then RUNTIME=singularity; else RUNTIME=apptainer; fi
fi

RUNG="$(printf 'rung_%02d' "$RUNG_IDX")"
CHUNK="chunk_${RUNG_IDX}"
THREADS="${SLURM_CPUS_PER_TASK:-4}"

STAIR="$RUN_DIR/staircase"
CHUNKS="$RUN_DIR/chunks"
SEQS="$RUN_DIR/sequences"
RESULTS="$RUN_DIR/results.csv"
mkdir -p "$RUN_DIR/vram" "$RUN_DIR/logs" "$RUN_DIR/outputs"
LOG="$RUN_DIR/logs/${STAGE}_${RUNG}.log"
VRAM_LOG="$RUN_DIR/vram/${STAGE}_${RUNG}.log"

# ---- seq_len from rungs.csv -------------------------------------------------
SEQ_LEN="$(awk -F, -v r="$RUNG_IDX" 'NR>1 && $1==r {print $3}' "$STAIR/rungs.csv")"
SEQ_LEN="${SEQ_LEN:-NA}"

# ---- helpers ----------------------------------------------------------------
sampler_pid=""
start_sampler() {
  # Manual loop (portable across nvidia-smi versions). One line per GPU per
  # tick; we take the max over all lines => peak of the busiest GPU/time.
  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null
      sleep 0.25
    done ) > "$VRAM_LOG" 2>/dev/null &
  sampler_pid=$!
}
stop_sampler() { [[ -n "$sampler_pid" ]] && kill "$sampler_pid" 2>/dev/null; wait "$sampler_pid" 2>/dev/null || true; }

vram_peak_mib() {
  awk 'NF{v=$1+0; if(v>m)m=v} END{print (m>0)?m:"NA"}' "$VRAM_LOG" 2>/dev/null || echo NA
}

host_ram_peak_mib() {
  local f v
  for f in /sys/fs/cgroup/memory.peak \
           /sys/fs/cgroup/memory/memory.max_usage_in_bytes; do
    if [[ -r "$f" ]]; then
      v="$(cat "$f" 2>/dev/null)"
      [[ "$v" =~ ^[0-9]+$ ]] && { echo $(( v / 1024 / 1024 )); return; }
    fi
  done
  echo NA
}

classify_status() {  # classify_status EXIT_CODE LOGFILE
  local code="$1" logf="$2"
  if [[ "$code" == 0 ]]; then echo ok; return; fi
  if grep -qiE 'cuda (error )?out of memory|CUDA_ERROR_OUT_OF_MEMORY|torch\.cuda\.OutOfMemoryError' "$logf" 2>/dev/null; then
    echo oom; return
  fi
  # Host OOM: a child (e.g. colabfold's `mmseqs result2msa`) gets SIGKILL'd by
  # the cgroup OOM-killer, but the parent CLI re-raises it as a generic exit-1,
  # hiding the 137. Scan the log for the kill signature so >--mem rungs are
  # labeled oom-host (the real reason) instead of a vague "error".
  if grep -qiE 'Signals\.SIGKILL|signal 9|SIGKILL|oom-kill(er)?|Out of memory|\bKilled\b' "$logf" 2>/dev/null; then
    echo oom-host; return
  fi
  # 137 = 128+SIGKILL (cgroup OOM-killer); 124 = GNU timeout / SLURM TERM
  case "$code" in
    137|139) echo oom-host ;;
    124|143) echo timeout ;;
    *)       echo error ;;
  esac
}

infer_s_from_log() {  # sum BENCH_INFER_S markers; blank if none
  awk '/^BENCH_INFER_S /{s+=$3; n++} END{if(n>0) printf "%.4f", s}' "$LOG" 2>/dev/null
}

msa_depth_from_a3m() {  # count sequences (> lines) in the rung's a3m, else NA
  local a3m
  a3m="$(ls "$SEQS/$RUNG"/msa/*.a3m 2>/dev/null | head -1)"
  [[ -n "$a3m" ]] && grep -c '^>' "$a3m" 2>/dev/null || echo NA
}

append_row() {  # append_row STATUS WALL INFER VRAM RAM MSADEPTH VARIANT
  local row="$STAGE,$7,$GPU_TYPE,$RUNG_IDX,$SEQ_LEN,$6,$1,$2,$3,$4,$5"
  local hdr="stage,variant,gpu_type,rung_idx,seq_len,msa_depth,status,wall_total_s,infer_s,vram_peak_mib,host_ram_peak_mib"
  if command -v flock >/dev/null 2>&1; then
    # flock (Linux) serializes concurrent array-task writes to the shared CSV.
    ( flock 9
      [[ -s "$RESULTS" ]] || echo "$hdr" >> "$RESULTS"
      echo "$row" >> "$RESULTS"
    ) 9>"$RUN_DIR/.results.lock"
  else
    # No flock (e.g. macOS dev): best-effort append. Single appends of one
    # line are near-atomic; collect.py can de-dup/sort afterwards if needed.
    [[ -s "$RESULTS" ]] || echo "$hdr" >> "$RESULTS"
    echo "$row" >> "$RESULTS"
  fi
}

# ---- stage dispatch: build the singularity exec -----------------------------
run_exec() {
  case "$STAGE" in
    msa)
      local sif db combined outdir
      sif="$(cfg containers.colabfold)" || { echo "no containers.colabfold in config" >&2; return 99; }
      db="$(cfg msa.mmseq2_db)"        || { echo "no msa.mmseq2_db in config" >&2; return 99; }
      combined="$CHUNKS/$CHUNK/combined.fasta"
      outdir="$RUN_DIR/colabfold/$RUNG"
      [[ -f "$combined" ]] || { echo "missing $combined (run chunk_fastas first)" >&2; return 99; }
      rm -rf "$outdir"; mkdir -p "$outdir/tmp"
      "$RUNTIME" exec --nv \
        -B "$combined:$combined" -B "$db:$db" -B "$outdir:$outdir" \
        "$sif" colabfold_search "$combined" "$db" "$outdir" --thread "$THREADS" --gpu 1
      ;;
    boltz)
      local sif cache yaml seqdir outdir
      sif="$(cfg containers.boltz)"   || { echo "no containers.boltz in config" >&2; return 99; }
      cache="$(cfg boltz.cache_dir)"  || { echo "no boltz.cache_dir in config" >&2; return 99; }
      seqdir="$SEQS/$RUNG"
      yaml="$seqdir/${RUNG}.yaml"
      outdir="$RUN_DIR/outputs/boltz/$RUNG"
      [[ -f "$yaml" ]] || { echo "missing $yaml (run msa stage first)" >&2; return 99; }
      mkdir -p "$outdir"
      "$RUNTIME" exec --nv \
        -B "$seqdir:$seqdir" -B "$outdir:$outdir" -B "$cache:/weights" \
        "$sif" boltz predict "$yaml" --cache /weights --out_dir "$outdir" \
          --devices 1 --accelerator gpu --override
      ;;
    esmfold)
      _run_esm_runner run_batch_esmfold.py containers.esmfold esmfold.cache_dir ""
      ;;
    esmc_300M|esmc_600M|esmc_6B)
      _run_esm_runner run_batch_esmc.py containers.esmc esmc.cache_dir "${STAGE#esmc_}"
      ;;
    *)
      echo "unknown stage: $STAGE" >&2; return 99 ;;
  esac
}

_run_esm_runner() {  # _run_esm_runner RUNNER.py SIF_KEY CACHE_KEY SIZE_OR_EMPTY
  local runner="$1" sif_key="$2" cache_key="$3" size="$4"
  local sif cache indir outdir runner_host
  sif="$(cfg "$sif_key")"    || { echo "no $sif_key in config" >&2; return 99; }
  cache="$(cfg "$cache_key")"|| { echo "no $cache_key in config" >&2; return 99; }
  runner_host="$CONTAINERS/$runner"
  indir="$SEQS/$RUNG"                 # holds only ${RUNG}.yaml (+ msa/, ignored by runner)
  outdir="$RUN_DIR/outputs/$STAGE/$RUNG"
  [[ -f "$indir/${RUNG}.yaml" ]] || { echo "missing $indir/${RUNG}.yaml (run msa stage first)" >&2; return 99; }
  [[ -f "$runner_host" ]] || { echo "missing runner $runner_host" >&2; return 99; }
  mkdir -p "$outdir"
  local size_flag=""
  [[ -n "$size" ]] && size_flag="--size $size"
  "$RUNTIME" exec --nv --cleanenv \
    --env HF_HOME=/models/hf --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    -B "$cache:/models/hf:ro" -B "$indir:/data/input:ro" -B "$outdir:/data/output" \
    -B "$runner_host:/opt/$runner:ro" \
    "$sif" python "/opt/$runner" \
      --cache /models/hf --input-dir /data/input --output-dir /data/output $size_flag
}

# ---- run --------------------------------------------------------------------
VARIANT="$STAGE"
case "$STAGE" in esmc_*) VARIANT="${STAGE#esmc_}" ;; esac

echo "[bench_one] stage=$STAGE $RUNG seq_len=$SEQ_LEN gpu=$GPU_TYPE runtime=$RUNTIME" | tee "$LOG"
start_sampler
t0=$(date +%s.%N)
run_exec >>"$LOG" 2>&1
exit_code=$?
t1=$(date +%s.%N)
stop_sampler

WALL=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
VRAM=$(vram_peak_mib)
RAM=$(host_ram_peak_mib)
STATUS=$(classify_status "$exit_code" "$LOG")

INFER=$(infer_s_from_log)
[[ -z "$INFER" ]] && INFER="$WALL"   # msa/boltz: no inner timer -> wall

case "$STAGE" in
  msa|boltz) MSADEPTH=$(msa_depth_from_a3m) ;;
  *)         MSADEPTH=NA ;;
esac

# MSA success: mint the YAML (+ absolute msa path) for downstream stages.
if [[ "$STAGE" == msa && "$exit_code" == 0 ]]; then
  python3 "$WF_SCRIPTS/organize_msa_outputs.py" \
    --file_list "$CHUNKS/$CHUNK/file_list.txt" \
    --colabfold_output "$RUN_DIR/colabfold/$RUNG" \
    --sequences_dir "$SEQS" >>"$LOG" 2>&1 || echo "[bench_one] WARN organize_msa_outputs failed" | tee -a "$LOG"
  python3 "$REPO_ROOT/scripts/fix_yaml_msa_paths.py" "$SEQS" >>"$LOG" 2>&1 || true
  MSADEPTH=$(msa_depth_from_a3m)
fi

append_row "$STATUS" "$WALL" "$INFER" "$VRAM" "$RAM" "$MSADEPTH" "$VARIANT"
echo "[bench_one] -> status=$STATUS wall=${WALL}s infer=${INFER}s vram=${VRAM}MiB ram=${RAM}MiB depth=$MSADEPTH" | tee -a "$LOG"

# Exit 0 even on OOM: the OOM rung is a recorded data point, not a job failure,
# so the SLURM array keeps marching up the ladder.
exit 0
