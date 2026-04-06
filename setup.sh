#!/bin/bash
set -euo pipefail

# ProtForge Setup Script
# ======================
# Two modes:
#   kempner  — Kempner cluster users: shared envs/weights already exist,
#              just generates config.yaml and validates paths.
#   custom   — Other SLURM clusters: creates conda envs, downloads models,
#              patches ESM, then generates config.
#
# Usage:
#   bash setup.sh                    # Auto-detects Kempner or prompts
#   bash setup.sh --mode kempner     # Kempner cluster (fast, no installs)
#   bash setup.sh --mode custom      # Other clusters (full install)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ===================== Kempner shared resources =====================
# These are pre-installed on the Kempner cluster shared filesystem.
# Kempner mode uses these directly; custom mode ignores them.
KEMPNER_MMSEQ2_DB="/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db"
KEMPNER_COLABFOLD_DB="/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/colabfold_db"
KEMPNER_COLABFOLD_BIN="/n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz/localcolabfold/colabfold-conda/bin"
KEMPNER_BOLTZ_CACHE="/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/boltz_db"
KEMPNER_BOLTZ_ENV="/n/holylfs06/LABS/kempner_shared/Everyone/common_envs/miniconda3/envs/boltz"
KEMPNER_SHARED_BASE="/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge"
KEMPNER_ESM_ENV="${KEMPNER_SHARED_BASE}/envs/esm"
KEMPNER_ESM_MODELS="${KEMPNER_SHARED_BASE}/esm_models_cache"
KEMPNER_ES_ENV="${KEMPNER_SHARED_BASE}/envs/es-analysis"
KEMPNER_PDANALYSIS="${KEMPNER_SHARED_BASE}/PDAnalysis"
KEMPNER_PARTITION="kempner_requeue"

# ===================== Parse arguments ==============================
MODE=""
SLURM_ACCOUNT=""
SLURM_EMAIL=""
FASTA_DIR=""
OUTPUT_DIR=""
SHARED_BASE=""

usage() {
    echo "Usage: bash setup.sh [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  --mode kempner       Kempner cluster (shared envs, no installs)"
    echo "  --mode custom        Other SLURM clusters (full install)"
    echo ""
    echo "Options:"
    echo "  --shared-base DIR    Base dir for envs/models (custom mode only)"
    echo "  --account ACCOUNT    SLURM account (required)"
    echo "  --email EMAIL        Email for SLURM notifications"
    echo "  --fasta-dir DIR      Input FASTA directory"
    echo "  --output-dir DIR     Output directory"
    echo "  -h, --help           Show this help"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)         MODE="$2"; shift 2 ;;
        --shared-base)  SHARED_BASE="$2"; shift 2 ;;
        --account)      SLURM_ACCOUNT="$2"; shift 2 ;;
        --email)        SLURM_EMAIL="$2"; shift 2 ;;
        --fasta-dir)    FASTA_DIR="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ===================== Detect / select mode =========================
echo "=============================================="
echo " ProtForge Setup"
echo "=============================================="
echo ""

if [[ -z "$MODE" ]]; then
    # Auto-detect: check if Kempner shared paths exist
    if [[ -d "$KEMPNER_SHARED_BASE" && -d "$KEMPNER_BOLTZ_ENV" ]]; then
        echo "Detected Kempner cluster (shared resources found)."
        read -rp "Use Kempner shared setup? [Y/n]: " yn
        yn="${yn:-Y}"
        if [[ "$yn" =~ ^[Yy] ]]; then
            MODE="kempner"
        else
            MODE="custom"
        fi
    else
        echo "Kempner shared paths not found — using custom setup."
        MODE="custom"
    fi
fi

echo "Mode: $MODE"
echo ""

# ===================== Common prompts ===============================
if [[ -z "$SLURM_ACCOUNT" ]]; then
    read -rp "SLURM account (e.g., kempner_bsabatini_lab): " SLURM_ACCOUNT
    if [[ -z "$SLURM_ACCOUNT" ]]; then
        echo "ERROR: SLURM account is required" >&2
        exit 1
    fi
fi

if [[ -z "$SLURM_EMAIL" ]]; then
    read -rp "Email for job notifications (optional): " SLURM_EMAIL
fi

if [[ -z "$FASTA_DIR" ]]; then
    read -rp "Input FASTA directory: " FASTA_DIR
    if [[ -z "$FASTA_DIR" ]]; then
        echo "WARNING: No FASTA dir set — you'll need to edit config.yaml later"
        FASTA_DIR="/path/to/your/fastas"
    fi
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    DEFAULT_OUTPUT="$HOME/outputs"
    read -rp "Output directory [$DEFAULT_OUTPUT]: " OUTPUT_DIR
    OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT}"
fi

LOG_DIR="$HOME/job_logs"

# ===================== Snakemake environment ========================
echo "=============================================="
echo " Snakemake environment"
echo "=============================================="

SNAKEMAKE_ACTIVATE=""

if command -v snakemake &>/dev/null; then
    echo "[OK] Snakemake already on PATH: $(which snakemake) ($(snakemake --version))"
else
    # Try to find an existing snakemake env
    SNAKEMAKE_FOUND=false
    for candidate in "$HOME/envs/snakemake" "$HOME/.conda/envs/snakemake"; do
        if [[ -x "$candidate/bin/snakemake" ]]; then
            echo "[OK] Found snakemake env at $candidate"
            SNAKEMAKE_ACTIVATE="conda activate $candidate"
            SNAKEMAKE_FOUND=true
            break
        fi
    done

    # Check if it's a named conda env
    if [[ "$SNAKEMAKE_FOUND" == false ]] && conda env list 2>/dev/null | grep -q "snakemake"; then
        echo "[OK] Found 'snakemake' conda environment"
        SNAKEMAKE_ACTIVATE="mamba activate snakemake"
        SNAKEMAKE_FOUND=true
    fi

    if [[ "$SNAKEMAKE_FOUND" == false ]]; then
        echo "[...] No snakemake environment found. Creating one..."
        SNAKEMAKE_ENV="$HOME/envs/snakemake"
        conda create -p "$SNAKEMAKE_ENV" python=3.12 -y
        "$SNAKEMAKE_ENV/bin/pip" install snakemake snakemake-executor-plugin-slurm
        SNAKEMAKE_ACTIVATE="conda activate $SNAKEMAKE_ENV"
        echo "[OK] Snakemake environment created at $SNAKEMAKE_ENV"
    fi
fi
echo ""

# ===================== Mode: Kempner ================================
if [[ "$MODE" == "kempner" ]]; then

    # All paths point to pre-existing shared resources
    MMSEQ2_DB="$KEMPNER_MMSEQ2_DB"
    COLABFOLD_DB="$KEMPNER_COLABFOLD_DB"
    COLABFOLD_BIN="$KEMPNER_COLABFOLD_BIN"
    BOLTZ_CACHE="$KEMPNER_BOLTZ_CACHE"
    BOLTZ_ENV="$KEMPNER_BOLTZ_ENV"
    ESM_ENV="$KEMPNER_ESM_ENV"
    ESM_MODELS="$KEMPNER_ESM_MODELS"
    ES_ENV="$KEMPNER_ES_ENV"
    PDANALYSIS_DIR="$KEMPNER_PDANALYSIS"
    SLURM_PARTITION="$KEMPNER_PARTITION"

    ENABLE_MSA=true
    ENABLE_BOLTZ=true

    echo "Using Kempner shared resources — no installation needed."
    echo ""

# ===================== Mode: Custom =================================
elif [[ "$MODE" == "custom" ]]; then

    # Prompt for shared base
    if [[ -z "$SHARED_BASE" ]]; then
        read -rp "Shared base directory for envs/models: " SHARED_BASE
        if [[ -z "$SHARED_BASE" ]]; then
            echo "ERROR: --shared-base is required for custom mode" >&2
            exit 1
        fi
    fi

    SLURM_PARTITION=""
    if [[ -z "$SLURM_PARTITION" ]]; then
        read -rp "Default SLURM partition: " SLURM_PARTITION
    fi

    ESM_ENV="${SHARED_BASE}/envs/esm"
    ESM_MODELS="${SHARED_BASE}/esm_models_cache"
    ES_ENV="${SHARED_BASE}/envs/es-analysis"
    PDANALYSIS_DIR="${SHARED_BASE}/PDAnalysis"

    # --- Prompt for external resource paths ---
    echo ""
    echo "External resource paths (leave blank to skip stage):"
    read -rp "  MMseqs2 database path: " MMSEQ2_DB
    read -rp "  ColabFold database path: " COLABFOLD_DB
    read -rp "  ColabFold binary path: " COLABFOLD_BIN
    read -rp "  Boltz model cache path: " BOLTZ_CACHE
    read -rp "  Boltz conda env path: " BOLTZ_ENV

    MMSEQ2_DB="${MMSEQ2_DB:-}"
    COLABFOLD_DB="${COLABFOLD_DB:-}"
    COLABFOLD_BIN="${COLABFOLD_BIN:-}"
    BOLTZ_CACHE="${BOLTZ_CACHE:-}"
    BOLTZ_ENV="${BOLTZ_ENV:-}"

    # Determine which stages can run based on provided paths
    if [[ -z "$MMSEQ2_DB" || -z "$COLABFOLD_DB" || -z "$COLABFOLD_BIN" ]]; then
        echo ""
        echo "[INFO] MSA resources not provided — pipeline.msa will be disabled"
        ENABLE_MSA=false
    else
        ENABLE_MSA=true
    fi

    if [[ -z "$BOLTZ_CACHE" || -z "$BOLTZ_ENV" ]]; then
        echo "[INFO] Boltz resources not provided — pipeline.boltz will be disabled"
        ENABLE_BOLTZ=false
    else
        ENABLE_BOLTZ=true
    fi

    echo ""
    echo "=============================================="
    echo " Creating environments"
    echo "=============================================="

    mkdir -p "$SHARED_BASE/envs"

    # --- ESM environment ---
    if [[ -d "$ESM_ENV" ]]; then
        echo "[OK] ESM env already exists at $ESM_ENV"
    else
        echo "[...] Creating ESM conda environment at $ESM_ENV"
        conda create -p "$ESM_ENV" python=3.12 -y
        "$ESM_ENV/bin/pip" install esm torch numpy pandas PyYAML tqdm
        echo "[OK] ESM environment created"
    fi

    # Verify ESM import
    if "$ESM_ENV/bin/python" -c "from esm.models.esmc import ESMC; print('[OK] ESM imports correctly')" 2>/dev/null; then
        :
    else
        echo "[ERROR] ESM package import failed." >&2
        exit 1
    fi

    # --- ES environment ---
    if [[ -d "$ES_ENV" ]]; then
        echo "[OK] ES env already exists at $ES_ENV"
    else
        echo "[...] Creating ES conda environment at $ES_ENV"
        conda create -p "$ES_ENV" python=3.12 -y
        "$ES_ENV/bin/pip" install numpy scipy pandas MDAnalysis
        echo "[OK] ES environment created"
    fi

    # --- PDAnalysis ---
    if [[ -d "$PDANALYSIS_DIR" ]]; then
        echo "[OK] PDAnalysis already exists at $PDANALYSIS_DIR"
    else
        echo "[...] Cloning PDAnalysis"
        git clone https://github.com/mirabdi/PDAnalysis "$PDANALYSIS_DIR"
        (cd "$PDANALYSIS_DIR" && "$ES_ENV/bin/pip" install -e .)
        echo "[OK] PDAnalysis installed"
    fi

    echo ""
    echo "=============================================="
    echo " Downloading ESM model weights"
    echo "=============================================="

    if [[ -f "$ESM_MODELS/esmc-600m-2024-12/data/weights/esmc_600m_2024_12_v0.pth" ]]; then
        echo "[OK] ESM weights already at $ESM_MODELS"
    else
        echo "[...] Downloading ESM model weights to $ESM_MODELS"
        mkdir -p "$ESM_MODELS"
        TORCH_HOME="$ESM_MODELS" HF_HOME="$ESM_MODELS" \
            "$ESM_ENV/bin/python" -c "
from esm.models.esmc import ESMC
ESMC.from_pretrained('esmc_600m')
print('[OK] ESM model downloaded')
"
    fi

    echo ""
    echo "=============================================="
    echo " Patching ESM hardcoded model paths"
    echo "=============================================="

    # Find the ESM constants file (handles different Python versions)
    ESM_CONSTANTS=$(find "$ESM_ENV/lib" -path "*/esm/utils/constants/esm3.py" -type f 2>/dev/null | head -1 || true)

    if [[ -z "$ESM_CONSTANTS" ]]; then
        echo "[WARNING] Could not find esm/utils/constants/esm3.py — skipping patch"
        echo "          You may need to manually update model paths in the ESM package."
    elif grep -q "$ESM_MODELS" "$ESM_CONSTANTS" 2>/dev/null; then
        echo "[OK] ESM paths already point to $ESM_MODELS"
    else
        echo "[...] Updating hardcoded paths in $ESM_CONSTANTS"
        CURRENT_CACHE=$(grep -oP 'Path\("\K[^"]+(?=/esmc-)' "$ESM_CONSTANTS" | head -1 || true)
        if [[ -n "$CURRENT_CACHE" ]]; then
            sed -i.bak "s|${CURRENT_CACHE}|${ESM_MODELS}|g" "$ESM_CONSTANTS"
            # Clear .pyc cache
            find "$(dirname "$ESM_CONSTANTS")/__pycache__" -name "*.pyc" -delete 2>/dev/null || true
            echo "[OK] Replaced '$CURRENT_CACHE' -> '$ESM_MODELS'"
        else
            echo "[WARNING] Could not detect hardcoded path. Manually edit:"
            echo "          $ESM_CONSTANTS"
        fi
    fi

    # Verify model loads
    echo "[...] Verifying ESM model loads..."
    if "$ESM_ENV/bin/python" -c "from esm.models.esmc import ESMC; ESMC.from_pretrained('esmc_600m'); print('[OK] Model loads correctly')" 2>/dev/null; then
        :
    else
        echo "[ERROR] ESM model failed to load after patching." >&2
        echo "        Expected: $ESM_MODELS/esmc-600m-2024-12/data/weights/esmc_600m_2024_12_v0.pth"
        exit 1
    fi
    echo ""

else
    echo "ERROR: Unknown mode '$MODE'. Use 'kempner' or 'custom'." >&2
    exit 1
fi

# ===================== Generate config.yaml =========================
echo "=============================================="
echo " Generating config.yaml"
echo "=============================================="

CONFIG_OUT="${SCRIPT_DIR}/config.yaml"
if [[ -f "$CONFIG_OUT" ]]; then
    BACKUP="${CONFIG_OUT}.bak.$(date +%Y%m%d_%H%M%S)"
    cp "$CONFIG_OUT" "$BACKUP"
    echo "[INFO] Existing config backed up to $BACKUP"
fi

EMAIL_LINE=""
if [[ -n "$SLURM_EMAIL" ]]; then
    EMAIL_LINE="  email: $SLURM_EMAIL"
else
    EMAIL_LINE="  # email: your@email.com  # Uncomment for SLURM notifications"
fi

# Helper: quote empty values for YAML safety (empty -> "", non-empty -> value as-is)
yaml_val() { if [[ -z "$1" ]]; then echo '""'; else echo "$1"; fi; }

cat > "$CONFIG_OUT" << YAML
# ProtForge Configuration
# Generated by setup.sh on $(date +%Y-%m-%d) (mode: ${MODE})

pipeline:
  msa: ${ENABLE_MSA}
  boltz: ${ENABLE_BOLTZ}
  esm: true
  es: false  # Enable after setting es.ref_path below

input:
  fasta_dir: ${FASTA_DIR}
  # yaml_dir: /path/to/yamls  # Use when pipeline.msa is false

output:
  parent_dir: ${OUTPUT_DIR}

# MSA generation
msa:
  max_files_per_job: 25
  array_max_concurrency: 10
  mmseq2_db: $(yaml_val "$MMSEQ2_DB")
  colabfold_db: $(yaml_val "$COLABFOLD_DB")
  colabfold_bin: $(yaml_val "$COLABFOLD_BIN")

# Boltz structure prediction
boltz:
  max_files_per_job: 25
  array_max_concurrency: 10
  delete_msa_after_processing: false
  recycling_steps: 10
  diffusion_samples: 25
  num_runs: 1
  cache_dir: $(yaml_val "$BOLTZ_CACHE")
  colabfold_db: $(yaml_val "$COLABFOLD_DB")
  env_path: $(yaml_val "$BOLTZ_ENV")

# ESM embeddings
esm:
  num_chunks: 1
  array_max_concurrency: 20
  env_path: ${ESM_ENV}
  cache_dir: ${ESM_MODELS}

# ES analysis (optional)
es:
  pdanalysis_dir: ${PDANALYSIS_DIR}
  ref_dir: ""
  ref_path: ""            # Set to your wildtype .cif file
  ref_seq: ""
  method: [strain]
  min_plddt: 70
  lddt_cutoffs: [0.125, 0.25, 0.5, 1]
  env_path: ${ES_ENV}

# SLURM settings
slurm:
  log_dir: ${LOG_DIR}
  partition: ${SLURM_PARTITION}
  account: ${SLURM_ACCOUNT}
${EMAIL_LINE}
YAML

echo "[OK] Config written to $CONFIG_OUT"
echo ""

# ===================== Create directories ===========================
mkdir -p "$LOG_DIR" 2>/dev/null || true
mkdir -p "$OUTPUT_DIR" 2>/dev/null || true

# ===================== Validate =====================================
echo "=============================================="
echo " Validation"
echo "=============================================="

ERRORS=0

check_path() {
    local label="$1" path="$2"
    if [[ -z "$path" ]]; then
        echo "  [--] $label — not configured (stage may be skipped)"
    elif [[ -e "$path" ]]; then
        echo "  [OK] $label"
    else
        echo "  [!!] $label — NOT FOUND: $path"
        ERRORS=$((ERRORS + 1))
    fi
}

check_path "MMseqs2 database"     "$MMSEQ2_DB"
check_path "ColabFold database"   "$COLABFOLD_DB"
check_path "ColabFold binaries"   "$COLABFOLD_BIN"
check_path "Boltz model cache"    "$BOLTZ_CACHE"
check_path "Boltz conda env"      "$BOLTZ_ENV"
check_path "ESM conda env"        "$ESM_ENV/bin/python"
check_path "ESM model weights"    "$ESM_MODELS/esmc-600m-2024-12/data/weights/esmc_600m_2024_12_v0.pth"
check_path "ES conda env"         "$ES_ENV/bin/python"
check_path "PDAnalysis"           "$PDANALYSIS_DIR"
check_path "Snakemake profile"    "$SCRIPT_DIR/profiles/slurm/config.yaml"

if command -v snakemake &>/dev/null; then
    echo "  [OK] Snakemake $(snakemake --version)"
elif [[ -n "$SNAKEMAKE_ACTIVATE" ]]; then
    echo "  [OK] Snakemake available (run: $SNAKEMAKE_ACTIVATE)"
else
    echo "  [!!] Snakemake not found"
    ERRORS=$((ERRORS + 1))
fi

echo ""
if [[ $ERRORS -eq 0 ]]; then
    echo "=============================================="
    echo " Setup complete! All checks passed."
    echo "=============================================="
    echo ""
    echo "Next steps:"
    STEP=1
    if [[ -n "$SNAKEMAKE_ACTIVATE" ]]; then
        echo "  $STEP. Activate snakemake:  $SNAKEMAKE_ACTIVATE"
        STEP=$((STEP + 1))
    fi
    echo "  $STEP. Review config.yaml"; STEP=$((STEP + 1))
    echo "  $STEP. Place your input FASTAs in: $FASTA_DIR"; STEP=$((STEP + 1))
    echo "  $STEP. Dry run:  snakemake --profile profiles/slurm/ -n"; STEP=$((STEP + 1))
    echo "  $STEP. Launch:   snakemake --profile profiles/slurm/"
else
    echo "=============================================="
    echo " Setup complete with $ERRORS warning(s)."
    echo " Fix the issues above before running."
    echo "=============================================="
fi
