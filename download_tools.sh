#!/bin/bash
set -eu

# Usage: ./download_tools.sh [--cache-dir DIR] [--config CONFIG_FILE] TOOL [TOOL ...]
# TOOL: boltz | es | esm
# --cache-dir: base dir for installs and model caches (default: ./protforge_cache)
# --config: read cache/install paths from config.yaml (overrides --cache-dir for those keys)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CACHE_BASE=""
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-dir)
      CACHE_BASE="${2:?Usage: --cache-dir requires DIR}"
      shift 2
      ;;
    --config)
      CONFIG_FILE="${2:?Usage: --config requires CONFIG_FILE}"
      shift 2
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

TOOLS=("$@")
if [[ ${#TOOLS[@]} -eq 0 ]]; then
  echo "Usage: $0 [--cache-dir DIR] [--config CONFIG_FILE] TOOL [TOOL ...]"
  echo "  TOOL: boltz | es | esm"
  echo "  --cache-dir: base dir for installs and model caches (default: ./protforge_cache)"
  echo "  --config: use paths from config.yaml where defined"
  exit 1
fi

get_config() {
  local key="$1"
  [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]] && python3 "${SCRIPT_DIR}/slurm_scripts/parse_config.py" "$CONFIG_FILE" "$key" 2>/dev/null || echo ""
}

# Resolve cache/install base
if [[ -z "$CACHE_BASE" ]]; then
  CACHE_BASE="${SCRIPT_DIR}/protforge_cache"
fi
CACHE_BASE="$(realpath -m "$CACHE_BASE")"
mkdir -p "$CACHE_BASE"
echo "Cache/install base: $CACHE_BASE"

for tool in "${TOOLS[@]}"; do
  case "$tool" in
    boltz)
      echo "=============================================="
      echo "Installing Boltz"
      echo "=============================================="
      BOLTZ_DIR="${CACHE_BASE}/boltz"
      BOLTZ_CACHE="${CACHE_BASE}/boltz_db"
      if [[ -n "$(get_config "boltz.cache_dir")" ]]; then
        BOLTZ_CACHE="$(get_config "boltz.cache_dir")"
      fi
      if [[ ! -d "$BOLTZ_DIR" ]]; then
        git clone https://github.com/jwohlwend/boltz.git "$BOLTZ_DIR"
        (cd "$BOLTZ_DIR" && pip install -e .[cuda])
      else
        echo "Boltz repo already at $BOLTZ_DIR"
      fi
      mkdir -p "$BOLTZ_CACHE"
      echo "Boltz cache dir: $BOLTZ_CACHE (set boltz.cache_dir in config to use)"
      ;;
    es)
      echo "=============================================="
      echo "Installing ES (PDAnalysis)"
      echo "=============================================="
      ES_DIR="${CACHE_BASE}/PDAnalysis"
      if [[ -n "$(get_config "es.script_dir")" ]]; then
        ES_DIR="$(get_config "es.script_dir")"
      fi
      if [[ ! -d "$ES_DIR" ]]; then
        git clone https://github.com/mirabdi/PDAnalysis "$ES_DIR"
        (cd "$ES_DIR" && python setup.py install)
      else
        echo "PDAnalysis already at $ES_DIR"
      fi
      echo "ES script_dir: $ES_DIR (set es.script_dir in config to use)"
      ;;
    esm)
      echo "=============================================="
      echo "Setting up ESM cache and pre-downloading models (if possible)"
      echo "=============================================="
      ESM_CACHE="${CACHE_BASE}/esm"
      if [[ -n "$(get_config "esm.cache_dir")" ]]; then
        ESM_CACHE="$(get_config "esm.cache_dir")"
      fi
      mkdir -p "$ESM_CACHE"
      export TORCH_HOME="$ESM_CACHE"
      export HF_HOME="$ESM_CACHE"
      if python3 -c "
import sys
try:
    import esm
    print('ESM package found, pre-downloading esm2_t33_650M_UR50D...')
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    print('Done.')
except ImportError:
    print('Install esm (pip install fair-esm) then re-run to pre-download weights.', file=sys.stderr)
except Exception as e:
    print('Pre-download failed:', e, file=sys.stderr)
" 2>/dev/null; then
        :
      else
        echo "ESM cache dir: $ESM_CACHE (set esm.cache_dir and TORCH_HOME for jobs)"
      fi
      ;;
    *)
      echo "Unknown tool: $tool (supported: boltz, es, esm)" >&2
      exit 1
      ;;
  esac
done

echo "Done."
