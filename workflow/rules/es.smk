"""
ES (Evolutionary Scale / Effective Strain) Stage Rules
=======================================================
run_es_all (single SLURM job that loops over all sequences)

Computes deformation metrics (effective strain, etc.) for each sequence by
averaging across multiple Boltz prediction runs using PDAnalysis's
AverageProtein. All sequences processed in a single SLURM job.

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
    """Resolve reference protein config into compute_es.py arguments.

    Returns the CLI flag string for the reference protein.
    """
    ref_dir = ES_CFG.get("ref_dir", "")
    ref_path = ES_CFG.get("ref_path", "")
    ref_seq = ES_CFG.get("ref_seq", "")

    if ref_seq:
        # Use a sequence from this pipeline's output as reference
        return f"--ref_dir {SEQUENCES_DIR}/{ref_seq}/boltz"
    elif ref_dir:
        return f"--ref_dir {ref_dir}"
    elif ref_path:
        return f"--ref_path {ref_path}"
    else:
        raise ValueError(
            "ES config must specify one of: es.ref_dir, es.ref_path, or es.ref_seq"
        )


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
        ref_args = resolve_ref_args(),
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
            PYTHON_CMD="{params.container_cmd} --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 python"
            COMPUTE_ES="/opt/protforge/workflow/scripts/compute_es.py"
            PDA_DIR="/opt/pdanalysis"
        else
            module load python/3.12.8-fasrc01 gcc/14.2.0-fasrc01 || true
            set +u
            source "$(conda info --base)/etc/profile.d/conda.sh"
            if [ -n "{params.env_path}" ]; then
                conda activate {params.env_path}
            fi
            set -u
            PYTHON_CMD="python"
            COMPUTE_ES="{workflow.basedir}/workflow/scripts/compute_es.py"
            PDA_DIR="{params.pdanalysis_dir}"
        fi

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

            # Skip if no CIF files
            CIF_COUNT=$(find "$BOLTZ_DIR" -name "*.cif" 2>/dev/null | wc -l)
            if [ "$CIF_COUNT" -eq 0 ]; then
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

            echo "=== Processing $SEQ_NAME ($CIF_COUNT CIF files) ==="
            if $PYTHON_CMD "$COMPUTE_ES" \
                {params.ref_args} \
                --seq_dir "$BOLTZ_DIR" \
                --output "$OUTPUT_CSV" \
                --pdanalysis_dir "$PDA_DIR" \
                --method {params.method} \
                --min_plddt {params.min_plddt} \
                --lddt_cutoffs {params.lddt_cutoffs} \
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

        if [ "$FAILED" -gt 0 ]; then
            echo "WARNING: $FAILED sequences failed — check logs in {params.es_dir}/logs/"
        fi

        touch {output.done}
        """
