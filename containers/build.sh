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

if [[ -n "${PROTFORGE_ROOT:-}" ]]; then
    if [[ -z "${SINGULARITY_CACHEDIR:-}" ]]; then
        export SINGULARITY_CACHEDIR="${PROTFORGE_ROOT%/}/sing_cache"
    fi
    if [[ -z "${SINGULARITY_TMPDIR:-}" ]]; then
        export SINGULARITY_TMPDIR="${PROTFORGE_ROOT%/}/sing_tmp"
    fi
    mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"
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

mkdir -p "$(dirname "$OUT")"

case "$MODE" in
    from-def)
        CMD=("$SING" build --fakeroot "$OUT" "$DEF_FILE")
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
