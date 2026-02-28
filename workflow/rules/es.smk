"""
ES (Evolutionary Scale) Stage Rules
=====================================
build_cif_paths -> run_es_analysis
"""

ES_CFG      = config.get("es", {})
ES_DIR      = f"{OUTPUT}/es"

ES_PARTITION = SLURM_CFG.get("es", {}).get("partition", SLURM_CFG.get("partition", ""))
ES_ACCOUNT   = SLURM_CFG.get("account", "")


def es_input(wildcards):
    """Input for build_cif_paths: depend on Boltz completion if Boltz is enabled."""
    inputs = {}
    if RUN_BOLTZ:
        inputs["boltz_done"] = f"{OUTPUT}/.boltz_complete"
    return inputs


rule build_cif_paths:
    """Find all best-model CIF files and write their paths to es/paths.txt."""
    input:
        unpack(es_input),
    output:
        paths = f"{ES_DIR}/paths.txt",
    params:
        sequences_dir = SEQUENCES_DIR,
    resources:
        cpus_per_task = 1,
        mem_mb = 2000,
        runtime = 10,
        slurm_partition = ES_PARTITION,
        slurm_account = ES_ACCOUNT,
    run:
        import os
        from pathlib import Path

        seq_dir = Path(params.sequences_dir)
        es_dir = Path(ES_DIR)
        es_dir.mkdir(parents=True, exist_ok=True)

        cif_paths = []
        for boltz_dir in sorted(seq_dir.glob("*/boltz")):
            cif_files = sorted(boltz_dir.glob("*.cif"))
            if cif_files:
                # Already organized to highest model, just take the first
                cif_paths.append(str(cif_files[0].resolve()))

        with open(output.paths, "w") as f:
            for p in cif_paths:
                f.write(p + "\n")

        print(f"Found {len(cif_paths)} CIF files for ES analysis")


rule run_es_analysis:
    """Run PDAnalysis ES analysis using MPI."""
    input:
        paths = f"{ES_DIR}/paths.txt",
    output:
        done = f"{ES_DIR}/.done",
    params:
        script_dir = ES_CFG.get("script_dir", ""),
        wt_path = ES_CFG.get("wt_path", ""),
        output_dir = ES_CFG.get("output_dir", ES_DIR),
        env_path = ES_CFG.get("env_path", ""),
        container_cmd = container_cmd("pdanalysis"),
    resources:
        cpus_per_task = 1,
        mem_mb = 8000,
        runtime = 30,
        tasks = 16,
        nodes = 1,
        slurm_partition = ES_PARTITION,
        slurm_account = ES_ACCOUNT,
        slurm_extra = "'--gpus-per-node=1'",
    shell:
        """
        set -euo pipefail
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        mkdir -p {params.output_dir}

        if [ -n "{params.container_cmd}" ]; then
            srun --mpi=pmix {params.container_cmd} \
                --env OMP_NUM_THREADS=1 \
                --env MKL_NUM_THREADS=1 \
                --env OPENBLAS_NUM_THREADS=1 \
                python /opt/pdanalysis/main.py \
                    --parallel True \
                    --protA {params.wt_path} \
                    --path_list {input.paths} \
                    --out_dir {params.output_dir} \
                    --method all --min_plddt 70 \
                    --lddt_cutoffs 0.125 0.25 0.5 1
        else
            module load python/3.12.8-fasrc01 gcc/14.2.0-fasrc01 openmpi || true
            set +u
            source "$(conda info --base)/etc/profile.d/conda.sh"
            if [ -n "{params.env_path}" ]; then
                conda activate {params.env_path}
            fi
            set -u
            cd {params.script_dir}
            mpiexec -n 16 python main.py \
                --parallel True \
                --protA {params.wt_path} \
                --path_list {input.paths} \
                --out_dir {params.output_dir} \
                --method all --min_plddt 70 \
                --lddt_cutoffs 0.125 0.25 0.5 1
        fi

        touch {output.done}
        """
