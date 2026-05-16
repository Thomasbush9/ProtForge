#!/usr/bin/env bash
# Build / fetch the ProtForge GPU Singularity image.
#
# Two modes:
#   1. --from-def (default): build locally from containers/protforge-gpu.def
#      via `singularity build --fakeroot`. Works on a Kempner compute node if
#      fakeroot is permitted. Run from an INTERACTIVE allocation, not a login
#      node, per the Kempner handbook:
#        salloc -p test -t 4:00:00 --mem 32G --ntasks-per-node 4
#
#   2. --from-docker docker://OWNER/IMAGE:TAG: pull a pre-built image (e.g.
#      from GHCR) and convert to SIF. This is the Kempner-handbook-canonical
#      path — `singularity pull` works from any compute node.
#
# Usage:
#   bash containers/build.sh                                   # needs PROTFORGE_SIF_DIR or PROTFORGE_ROOT or -o
#   bash containers/build.sh -o /path/to/out.sif               # custom output path
#   bash containers/build.sh --from-docker docker://ghcr.io/me/protforge-gpu:latest
#   bash containers/build.sh --dry-run                         # print command without running
#
# Requires:
#   - singularity (Kempner uses `singularity`, not `apptainer`)
#   - For --from-def: --fakeroot support (try it; fall back to --from-docker if not)
#   - Network access for the first build/pull (~10 GB of weights + deps)
#
# Notes on iteration speed (--from-def mode):
#   `singularity build` doesn't cache %post layers. Every rebuild re-runs the
#   full %post, including the ~7 GB model download. Two ways to mitigate:
#
#   1. Sandbox dev loop (RECOMMENDED for iteration):
#        singularity build --sandbox /tmp/pfsandbox containers/protforge-gpu.def
#        # iterate on code, then:
#        singularity build protforge-gpu.sif /tmp/pfsandbox
#
#   2. Set SINGULARITY_CACHEDIR / SINGULARITY_TMPDIR somewhere with space, or
#      export PROTFORGE_ROOT — build.sh will default them under that tree when unset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
DEF_FILE="${SCRIPT_DIR}/protforge-gpu.def"

# Output SIF path (set by -o/--output in the arg loop, else resolved after the loop).
OUT=""
MODE="from-def"
DOCKER_URL=""
DRY_RUN=0
LOG_FILE=""        # set by --log; else auto-placed next to the SIF
LOG_DISABLED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            OUT="$2"; shift 2 ;;
        --from-def)
            MODE="from-def"; shift ;;
        --from-docker)
            MODE="from-docker"; DOCKER_URL="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --log)
            LOG_FILE="$2"; shift 2 ;;
        --no-log)
            LOG_DISABLED=1; shift ;;
        -h|--help)
            sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$OUT" ]]; then
    if [[ -n "${PROTFORGE_SIF_DIR:-}" ]]; then
        OUT="${PROTFORGE_SIF_DIR%/}/protforge-gpu.sif"
    elif [[ -n "${PROTFORGE_ROOT:-}" ]]; then
        OUT="${PROTFORGE_ROOT%/}/sifs/protforge-gpu.sif"
    else
        echo "ERROR: no output path for the SIF. Set one of:" >&2
        echo "  - pass -o/--output /path/to/protforge-gpu.sif" >&2
        echo "  - export PROTFORGE_SIF_DIR (writes \$PROTFORGE_SIF_DIR/protforge-gpu.sif)" >&2
        echo "  - export PROTFORGE_ROOT (writes \$PROTFORGE_ROOT/sifs/protforge-gpu.sif)" >&2
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
[[ -n "${SINGULARITY_TMPDIR:-}"   && -z "${APPTAINER_TMPDIR:-}"   ]] && export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"

# Auto-log the entire run to a timestamped file so users can share it with us
# on failure. Default location: <SING_BASE>/build-logs/build-<ts>.log if a
# base is known, else next to the SIF in <dirname OUT>/build-logs/. Override
# with --log /path; disable with --no-log.
if (( ! LOG_DISABLED )) && [[ -z "$LOG_FILE" ]]; then
    log_base=""
    if [[ -n "$SING_BASE" ]]; then
        log_base="${SING_BASE}/build-logs"
    else
        log_base="$(dirname "$OUT")/build-logs"
    fi
    mkdir -p "$log_base"
    LOG_FILE="${log_base}/build-$(date +%Y-%m-%dT%H-%M-%S).log"
fi
if (( ! LOG_DISABLED )); then
    mkdir -p "$(dirname "$LOG_FILE")"
    # Tee stdout+stderr from this point on into the log file. `process
    # substitution + exec` keeps the original tty output intact while
    # capturing everything (including singularity's own progress bars).
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

# Report which runtime + version we'll use, so test logs are self-documenting.
# (Apptainer's compat symlink at /usr/bin/singularity reports "apptainer
# version X.Y.Z" — distinguishes from SingularityCE's "singularity-ce ...".)
echo "Runtime     : $SING ($("$SING" --version 2>&1 | head -1))"

mkdir -p "$(dirname "$OUT")"

# Guard: refuse to build if SINGULARITY_TMPDIR or $OUT's parent lives under
# REPO_ROOT. The def file uses `%files . /opt/protforge`, which `cp -r`s the
# build context (REPO_ROOT) into the build's rootfs. If the rootfs lives
# inside REPO_ROOT, cp recurses into itself and dies with "cannot copy a
# directory into itself". This is *exactly* the foot-gun that happens when
# users set PROTFORGE_ROOT to the repo path instead of its parent.
if [[ "$MODE" == "from-def" ]]; then
    repo_real="$(cd "$REPO_ROOT" && pwd -P)"
    for p in "${SINGULARITY_TMPDIR:-}" "$(dirname "$OUT")"; do
        [[ -z "$p" ]] && continue
        # Resolve to an absolute path if it exists; otherwise leave as-is.
        if [[ -d "$p" ]]; then
            p_real="$(cd "$p" && pwd -P)"
        else
            p_real="$p"
        fi
        case "$p_real/" in
            "$repo_real"/*)
                echo "ERROR: $p resolves inside REPO_ROOT ($repo_real)." >&2
                echo "  This collides with '%files . /opt/protforge' in the def file" >&2
                echo "  ('cp: cannot copy a directory into itself'). PROTFORGE_ROOT must" >&2
                echo "  be the *parent* of the repo, not the repo path itself. See" >&2
                echo "  containers/README.md for the expected layout." >&2
                exit 1
                ;;
        esac
    done
fi

case "$MODE" in
    from-def)
        # --force overwrites an existing SIF at $OUT. Without it, re-running
        # after a failed/partial build aborts with "image file already exists".
        # Matches the --force behavior of the pull branch below.
        CMD=("$SING" build --force --fakeroot "$OUT" "$DEF_FILE")
        ;;
    from-docker)
        if [[ -z "$DOCKER_URL" ]]; then
            echo "ERROR: --from-docker requires a docker:// URL" >&2
            exit 2
        fi
        # `singularity pull` writes to its own filename; force --name to OUT.
        CMD=("$SING" pull --force --name "$OUT" "$DOCKER_URL")
        ;;
esac

echo "Mode        : $MODE"
echo "Build context: $REPO_ROOT"
[[ "$MODE" == "from-def" ]]    && echo "Def file    : $DEF_FILE"
[[ "$MODE" == "from-docker" ]] && echo "Docker URL  : $DOCKER_URL"
echo "Output SIF  : $OUT"
[[ -n "${SINGULARITY_CACHEDIR:-}" ]] && echo "SINGULARITY_CACHEDIR: $SINGULARITY_CACHEDIR"
[[ -n "${SINGULARITY_TMPDIR:-}" ]]   && echo "SINGULARITY_TMPDIR  : $SINGULARITY_TMPDIR"
echo "Command     : ${CMD[*]}"

if (( DRY_RUN )); then
    exit 0
fi

# Run build from repo root so %files paths resolve correctly.
cd "$REPO_ROOT"
"${CMD[@]}"

echo
echo "Done. Image at: $OUT"
echo "Size: $(du -h "$OUT" | cut -f1)"
[[ -n "$LOG_FILE" ]] && echo "Build log saved: $LOG_FILE"
