#!/usr/bin/env bash
# Build / fetch the ProtForge GPU Singularity images.
#
# ProtForge uses ONE image PER STAGE (not a single mega-SIF). Each GPU stage
# runs inside its own image, built from its own def file:
#
#   stage      def file          output SIF    serves
#   ---------  ----------------  ------------  ------------------------------
#   msa        msa.def           msa.sif       MSA (colabfold_search+mmseqs2)
#   boltz      boltz.def         boltz.sif     Boltz structure prediction
#   esm        esmfold_cu.def    esm.sif       ESM-C embeddings + ESMFold2
#   openfold   openfold.def      openfold.sif  OpenFold3
#
# After building, point config.yaml at the SIFs (see config.template.yaml):
#   containers:
#     colabfold: <sifdir>/msa.sif
#     boltz:     <sifdir>/boltz.sif
#     esmc:      <sifdir>/esm.sif       # esmc + esmfold share the ESM image
#     esmfold:   <sifdir>/esm.sif
#     openfold:  <sifdir>/openfold.sif
#
# Two modes:
#   1. --from-def (default): build locally via `singularity build --fakeroot`.
#      Works on a Kempner compute node if fakeroot is permitted. Run from an
#      INTERACTIVE allocation, NOT a login node, per the Kempner handbook:
#        salloc -p test -t 4:00:00 --mem 32G --ntasks-per-node 4
#
#   2. --from-docker docker://OWNER/IMAGE:TAG: pull a pre-built image and
#      convert to SIF. `singularity pull` works from any compute node. One URL
#      per stage, so --from-docker takes exactly one stage.
#
# Usage:
#   bash containers/build.sh all                     # build every stage image
#   bash containers/build.sh boltz                   # build one stage
#   bash containers/build.sh esm openfold            # build a subset
#   bash containers/build.sh boltz -o /path/boltz.sif
#   bash containers/build.sh boltz --from-docker docker://ghcr.io/you/protforge-boltz:latest
#   bash containers/build.sh boltz --dry-run         # print command without running
#
# Output dir (per-stage SIF names appended): -o/--output (single stage only),
# else $PROTFORGE_SIF_DIR, else $PROTFORGE_ROOT/sifs. Model weights and the big
# MSA/Boltz databases are NOT baked in — they are bind-mounted at runtime and
# fetched separately (scripts/download_models.py; shared DBs already on Kempner).
#
# Notes on iteration speed (--from-def mode):
#   `singularity build` doesn't cache %post layers — every rebuild re-runs the
#   full %post. To iterate on a def, use the sandbox loop:
#        singularity build --sandbox /tmp/pf_boltz containers/boltz.def
#        singularity build boltz.sif /tmp/pf_boltz
#   Or set SINGULARITY_CACHEDIR / SINGULARITY_TMPDIR somewhere with space (or
#   export PROTFORGE_ROOT — build.sh defaults them under that tree when unset).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stage -> def file + output SIF basename. `esm` builds the shared ESM image
# (esmfold_cu.def) used by both the esmc and esmfold config keys.
stage_def() {
  case "$1" in
  msa) echo "msa.def" ;;
  boltz) echo "boltz.def" ;;
  esm) echo "esmfold_cu.def" ;;
  openfold) echo "openfold.def" ;;
  *) return 1 ;;
  esac
}
stage_sif() {
  case "$1" in
  msa) echo "msa.sif" ;;
  boltz) echo "boltz.sif" ;;
  esm) echo "esm.sif" ;;
  openfold) echo "openfold.sif" ;;
  *) return 1 ;;
  esac
}
ALL_STAGES=(msa boltz esm openfold)

STAGES=()
OUT=""           # -o/--output: explicit SIF path (single stage only)
MODE="from-def"
DOCKER_URL=""
DRY_RUN=0
LOG_FILE=""
LOG_DISABLED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
  -o | --output)
    OUT="$2"
    shift 2
    ;;
  --from-def)
    MODE="from-def"
    shift
    ;;
  --from-docker)
    MODE="from-docker"
    DOCKER_URL="$2"
    shift 2
    ;;
  --dry-run)
    DRY_RUN=1
    shift
    ;;
  --log)
    LOG_FILE="$2"
    shift 2
    ;;
  --no-log)
    LOG_DISABLED=1
    shift
    ;;
  -h | --help)
    sed -n '2,45p' "${BASH_SOURCE[0]}"
    exit 0
    ;;
  all)
    STAGES=("${ALL_STAGES[@]}")
    shift
    ;;
  msa | boltz | esm | openfold)
    STAGES+=("$1")
    shift
    ;;
  *)
    echo "Unknown arg: $1" >&2
    echo "Stages: msa | boltz | esm | openfold | all. See --help." >&2
    exit 2
    ;;
  esac
done

if [[ ${#STAGES[@]} -eq 0 ]]; then
  echo "ERROR: no stage selected. Pass one or more of: msa boltz esm openfold (or 'all')." >&2
  echo "  e.g. bash containers/build.sh all" >&2
  exit 2
fi

# De-duplicate stages while preserving order (so 'all esm' isn't a double build).
_seen=" "
_dedup=()
for s in "${STAGES[@]}"; do
  [[ "$_seen" == *" $s "* ]] && continue
  _seen+="$s "
  _dedup+=("$s")
done
STAGES=("${_dedup[@]}")

if [[ -n "$OUT" && ${#STAGES[@]} -ne 1 ]]; then
  echo "ERROR: -o/--output only works with a single stage (got ${#STAGES[@]})." >&2
  exit 2
fi
if [[ "$MODE" == "from-docker" && ${#STAGES[@]} -ne 1 ]]; then
  echo "ERROR: --from-docker takes exactly one stage (one image URL per stage)." >&2
  exit 2
fi

# Resolve the output directory for stages that didn't get an explicit -o.
SIF_DIR=""
if [[ -z "$OUT" ]]; then
  if [[ -n "${PROTFORGE_SIF_DIR:-}" ]]; then
    SIF_DIR="${PROTFORGE_SIF_DIR%/}"
  elif [[ -n "${PROTFORGE_ROOT:-}" ]]; then
    SIF_DIR="${PROTFORGE_ROOT%/}/sifs"
  else
    echo "ERROR: no output directory for the SIFs. Set one of:" >&2
    echo "  - pass -o/--output /path/to/<stage>.sif   (single stage only)" >&2
    echo "  - export PROTFORGE_SIF_DIR  (writes \$PROTFORGE_SIF_DIR/<stage>.sif)" >&2
    echo "  - export PROTFORGE_ROOT     (writes \$PROTFORGE_ROOT/sifs/<stage>.sif)" >&2
    echo "Non-interactive bash does not load ~/.bashrc; export in the job script or use -o." >&2
    exit 1
  fi
fi

# Pick a base for SINGULARITY_CACHEDIR/TMPDIR defaults:
#   - PROTFORGE_ROOT if set (canonical case)
#   - else dirname of PROTFORGE_SIF_DIR (so users who only set SIF_DIR still get
#     big-FS cache/tmp dirs instead of /tmp, which is too small for fakeroot
#     builds and often noexec on compute nodes).
SING_BASE=""
if [[ -n "${PROTFORGE_ROOT:-}" ]]; then
  SING_BASE="${PROTFORGE_ROOT%/}"
elif [[ -n "${PROTFORGE_SIF_DIR:-}" ]]; then
  SING_BASE="$(dirname "${PROTFORGE_SIF_DIR%/}")"
elif [[ -n "$OUT" ]]; then
  SING_BASE="$(dirname "$(dirname "$OUT")")"
fi
if [[ -n "$SING_BASE" ]]; then
  if [[ -z "${SINGULARITY_CACHEDIR:-}" ]]; then
    export SINGULARITY_CACHEDIR="${SING_BASE}/sing_cache"
  fi
  if [[ -z "${SINGULARITY_TMPDIR:-}" ]]; then
    export SINGULARITY_TMPDIR="${SING_BASE}/sing_tmp"
  fi
  mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"
fi

# Apptainer (the LF fork) reads both APPTAINER_* and SINGULARITY_* names, but
# APPTAINER_* is preferred and SINGULARITY_* is marked legacy on recent
# releases (emits deprecation warnings). Export both so the same script works
# under either runtime without warnings.
[[ -n "${SINGULARITY_CACHEDIR:-}" && -z "${APPTAINER_CACHEDIR:-}" ]] && export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
[[ -n "${SINGULARITY_TMPDIR:-}" && -z "${APPTAINER_TMPDIR:-}" ]] && export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"

# Auto-log the entire run to a timestamped file so users can share it with us
# on failure. Default: <SING_BASE>/build-logs/, else next to the SIF dir.
# Override with --log /path; disable with --no-log.
if ((!LOG_DISABLED)) && [[ -z "$LOG_FILE" ]]; then
  if [[ -n "$SING_BASE" ]]; then
    log_base="${SING_BASE}/build-logs"
  elif [[ -n "$SIF_DIR" ]]; then
    log_base="${SIF_DIR}/build-logs"
  else
    log_base="$(dirname "$OUT")/build-logs"
  fi
  mkdir -p "$log_base"
  LOG_FILE="${log_base}/build-$(date +%Y-%m-%dT%H-%M-%S).log"
fi
if ((!LOG_DISABLED)); then
  mkdir -p "$(dirname "$LOG_FILE")"
  # Tee stdout+stderr from here on into the log file while keeping tty output.
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo "Logging to  : $LOG_FILE"
  echo "             (rerun with --no-log to disable, or --log PATH to override)"
fi

if ! command -v singularity >/dev/null 2>&1; then
  if command -v apptainer >/dev/null 2>&1; then
    SING=apptainer
  else
    echo "ERROR: neither singularity nor apptainer is on PATH" >&2
    exit 1
  fi
else
  SING=singularity
fi

# Report which runtime + version we'll use, so build logs are self-documenting.
echo "Runtime     : $SING ($("$SING" --version 2>&1 | head -1))"
echo "Stages      : ${STAGES[*]}"
echo "Mode        : $MODE"
[[ -n "$SIF_DIR" ]] && echo "Output dir  : $SIF_DIR"
[[ -n "${SINGULARITY_CACHEDIR:-}" ]] && echo "SINGULARITY_CACHEDIR: $SINGULARITY_CACHEDIR"
[[ -n "${SINGULARITY_TMPDIR:-}" ]] && echo "SINGULARITY_TMPDIR  : $SINGULARITY_TMPDIR"

build_one() {
  local stage="$1"
  local def out
  def="${SCRIPT_DIR}/$(stage_def "$stage")"
  if [[ -n "$OUT" ]]; then
    out="$OUT"
  else
    out="${SIF_DIR}/$(stage_sif "$stage")"
  fi
  mkdir -p "$(dirname "$out")"

  local cmd
  case "$MODE" in
  from-def)
    if [[ ! -f "$def" ]]; then
      echo "ERROR: def file not found: $def" >&2
      return 1
    fi
    # --force overwrites an existing/partial SIF (matches the pull branch).
    cmd=("$SING" build --force --fakeroot "$out" "$def")
    ;;
  from-docker)
    cmd=("$SING" pull --force --name "$out" "$DOCKER_URL")
    ;;
  esac

  echo
  echo "=== stage: $stage ==="
  [[ "$MODE" == "from-def" ]] && echo "Def file    : $def"
  [[ "$MODE" == "from-docker" ]] && echo "Docker URL  : $DOCKER_URL"
  echo "Output SIF  : $out"
  echo "Command     : ${cmd[*]}"

  if ((DRY_RUN)); then
    return 0
  fi

  "${cmd[@]}"
  echo "Done. Image at: $out"
  echo "Size: $(du -h "$out" | cut -f1)"

  # Record the image digest as a `<sif>.sha256` sidecar. The per-run provenance
  # manifest reads this instead of re-hashing multi-GB SIFs on every launch —
  # this is what keeps a run auditable even though the def files build from
  # floating base-image tags.
  if command -v sha256sum >/dev/null 2>&1; then
    echo "Computing sha256 digest (one-time, may take a moment)…"
    (cd "$(dirname "$out")" && sha256sum "$(basename "$out")" >"$(basename "$out").sha256")
    echo "Digest      : $(cut -d' ' -f1 "$out.sha256")"
  else
    echo "WARN: sha256sum not found; skipping digest sidecar." >&2
  fi
}

rc=0
for stage in "${STAGES[@]}"; do
  if ! build_one "$stage"; then
    echo "ERROR: build failed for stage '$stage'." >&2
    rc=1
    break
  fi
done

echo
if ((rc == 0)); then
  echo "All requested stage images built: ${STAGES[*]}"
  [[ -n "$SIF_DIR" ]] && echo "Set the containers.* paths in config.yaml to the SIFs under: $SIF_DIR"
fi
[[ -n "$LOG_FILE" ]] && echo "Build log saved: $LOG_FILE"
exit "$rc"
