"""
ES (Evolutionary Scale / Effective Strain) Stage Rules
=======================================================
run_es_all (single SLURM job that loops over all sequences)

Computes deformation metrics (effective strain, etc.) for each sequence by
averaging across multiple Boltz prediction runs using PDAnalysis's
AverageProtein. All sequences processed in a single SLURM job.

Calls PDAnalysis main.py directly (no wrapper script) to avoid
subprocess python interpreter issues.

Reference protein options (from config):
  es.ref_dir:  path to boltz dir with run_N/ subdirs  -> AverageProtein
  es.ref_path: path to a single CIF file              -> Protein
  es.ref_seq:  sequence name in sequences/ dir         -> resolved to ref_dir
"""

ES_CFG = config.get("es", {})
ES_DIR = f"{OUTPUT}/es"

ES_PARTITION = SLURM_CFG.get("es", {}).get("partition", SLURM_CFG.get("partition", ""))
ES_ACCOUNT = SLURM_CFG.get("account", "")

# PDAnalysis directory (support both pdanalysis_dir and legacy script_dir)
PDANALYSIS_DIR = ES_CFG.get("pdanalysis_dir", ES_CFG.get("script_dir", ""))

# ES deformation parameters
ES_METHOD = ES_CFG.get("method", ["strain"])
ES_MIN_PLDDT = ES_CFG.get("min_plddt", 70)
ES_LDDT_CUTOFFS = ES_CFG.get("lddt_cutoffs", [0.125, 0.25, 0.5, 1])


def resolve_ref_args():
    """Resolve reference protein config into main.py --protA arguments.

    Returns (flag, value) where flag is --protA or empty for list mode.
    For ref_dir, returns the shell glob pattern to expand CIF files.
    For ref_path, returns the single file path.
    """
    ref_dir = ES_CFG.get("ref_dir", "")
    ref_path = ES_CFG.get("ref_path", "")
    ref_seq = ES_CFG.get("ref_seq", "")

    if ref_seq:
        return f"{SEQUENCES_DIR}/{ref_seq}/boltz"
    elif ref_dir:
        return ref_dir
    elif ref_path:
        return ref_path
    else:
        raise ValueError(
            "ES config must specify one of: es.ref_dir, es.ref_path, or es.ref_seq"
        )


def ref_is_single_file():
    """Check if reference is a single file (ref_path) vs directory."""
    ref_path = ES_CFG.get("ref_path", "")
    ref_dir = ES_CFG.get("ref_dir", "")
    ref_seq = ES_CFG.get("ref_seq", "")
    # ref_path takes effect only when ref_seq and ref_dir are empty
    return bool(ref_path) and not bool(ref_seq) and not bool(ref_dir)


def es_input(wildcards):
    """Input for run_es_all: depend on Boltz completion if enabled."""
    inputs = {}
    if RUN_BOLTZ:
        inputs["boltz_done"] = f"{OUTPUT}/.boltz_complete"
    return inputs


rule run_es_all:
    """Run PDAnalysis ES for all sequences in a single SLURM job."""
    input:
        unpack(es_input),
    output:
        done = f"{ES_DIR}/.done",
    log:
        f"{ES_DIR}/es_all.log",
    params:
        sequences_dir = SEQUENCES_DIR,
        es_dir = ES_DIR,
        pdanalysis_dir = PDANALYSIS_DIR,
        ref_source = resolve_ref_args(),
        ref_is_file = ref_is_single_file(),
        method = " ".join(ES_METHOD) if isinstance(ES_METHOD, list) else ES_METHOD,
        min_plddt = ES_MIN_PLDDT,
        lddt_cutoffs = " ".join(str(c) for c in ES_LDDT_CUTOFFS) if isinstance(ES_LDDT_CUTOFFS, list) else str(ES_LDDT_CUTOFFS),
        env_path = ES_CFG.get("env_path", ""),
        container_cmd = container_cmd("pdanalysis"),
    resources:
        cpus_per_task = 4,
        mem_mb = 16000,
        runtime = 120,
        slurm_partition = ES_PARTITION,
        slurm_account = ES_ACCOUNT,
        slurm_extra = "'--gpus-per-node=1'",
    shell:
        """
        set -euo pipefail
        exec > {log} 2>&1
        set -x
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1

        mkdir -p {params.es_dir}/logs

        if [ -n "{params.container_cmd}" ]; then
            PDA_MAIN="/opt/pdanalysis/main.py"
        else
            set +u
            eval "$(conda shell.bash hook)"
            if [ -n "{params.env_path}" ]; then
                conda activate {params.env_path}
            fi
            set -u
            echo "Using python: $(which python)"
            PDA_MAIN="{params.pdanalysis_dir}/main.py"
        fi

        # Helper: collect CIF files from a boltz dir (run_N/ subdirs or flat)
        collect_cifs() {{
            local DIR="$1"
            local CIFS=""
            if ls "$DIR"/run_*/*.cif 1>/dev/null 2>&1; then
                # Multi-run: pick highest model from each run_N/
                for RUN in $(ls -d "$DIR"/run_*/ 2>/dev/null | sort); do
                    CIF=$(ls "$RUN"/*_model_*.cif 2>/dev/null | sort | tail -1)
                    if [ -z "$CIF" ]; then
                        CIF=$(ls "$RUN"/*.cif 2>/dev/null | sort | tail -1)
                    fi
                    if [ -n "$CIF" ]; then
                        CIFS="$CIFS $CIF"
                    fi
                done
            else
                # Single-run: pick highest model CIF directly
                CIF=$(ls "$DIR"/*_model_*.cif 2>/dev/null | sort | tail -1)
                if [ -z "$CIF" ]; then
                    CIF=$(ls "$DIR"/*.cif 2>/dev/null | sort | tail -1)
                fi
                if [ -n "$CIF" ]; then
                    CIFS="$CIF"
                fi
            fi
            echo "$CIFS"
        }}

        # Build reference --protA args
        if [ "{params.ref_is_file}" = "True" ]; then
            REF_CIFS="{params.ref_source}"
        else
            REF_CIFS=$(collect_cifs "{params.ref_source}")
        fi

        if [ -z "$REF_CIFS" ]; then
            echo "ERROR: No reference CIF files found in {params.ref_source}"
            exit 1
        fi
        echo "Reference CIFs: $REF_CIFS"

        TOTAL=0
        SUCCESS=0
        FAILED=0

        for SEQ_DIR in {params.sequences_dir}/*/; do
            SEQ_NAME=$(basename "$SEQ_DIR")
            BOLTZ_DIR="${{SEQ_DIR}}boltz"

            # Skip if no boltz directory
            if [ ! -d "$BOLTZ_DIR" ]; then
                echo "SKIP $SEQ_NAME: no boltz dir"
                continue
            fi

            # Collect sequence CIF files
            SEQ_CIFS=$(collect_cifs "$BOLTZ_DIR")
            if [ -z "$SEQ_CIFS" ]; then
                echo "SKIP $SEQ_NAME: no CIF files"
                continue
            fi

            TOTAL=$((TOTAL + 1))
            OUTPUT_CSV="{params.es_dir}/${{SEQ_NAME}}.csv"

            # Skip if already computed
            if [ -f "$OUTPUT_CSV" ]; then
                echo "SKIP $SEQ_NAME: already computed"
                SUCCESS=$((SUCCESS + 1))
                continue
            fi

            echo "=== Processing $SEQ_NAME ==="
            if python "$PDA_MAIN" \
                --protA $REF_CIFS \
                --protB $SEQ_CIFS \
                --method {params.method} \
                --min_plddt {params.min_plddt} \
                --lddt_cutoffs {params.lddt_cutoffs} \
                -o "$OUTPUT_CSV" \
                > {params.es_dir}/logs/${{SEQ_NAME}}.log 2>&1; then
                echo "OK $SEQ_NAME"
                SUCCESS=$((SUCCESS + 1))
            else
                echo "FAILED $SEQ_NAME (see {params.es_dir}/logs/${{SEQ_NAME}}.log)"
                FAILED=$((FAILED + 1))
            fi
        done

        echo "=== ES Summary: $SUCCESS/$TOTAL succeeded, $FAILED failed ==="

        if [ "$TOTAL" -eq 0 ]; then
            echo "ERROR: No sequences found in {params.sequences_dir}"
            exit 1
        fi

        if [ "$SUCCESS" -eq 0 ]; then
            echo "ERROR: All $TOTAL sequences failed — check logs in {params.es_dir}/logs/"
            exit 1
        fi

        if [ "$FAILED" -gt 0 ]; then
            echo "WARNING: $FAILED sequences failed — check logs in {params.es_dir}/logs/"
        fi

        touch {output.done}
        """
