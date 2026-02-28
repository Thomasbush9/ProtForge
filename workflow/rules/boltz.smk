"""
Boltz Stage Rules
==================
checkpoint chunk_yamls_for_boltz -> run_boltz_predict (per chunk) ->
organize_boltz_chunk (per chunk) -> boltz_complete (aggregate)
"""

BOLTZ_CFG     = config.get("boltz", {})
BOLTZ_CHUNKS  = f"{OUTPUT}/boltz_chunks"

# Determine YAML source: from MSA stage (sequences/) or user-provided yaml_dir
YAML_SOURCE_DIR = f"{OUTPUT}/sequences" if RUN_MSA else config["input"].get("yaml_dir", "")

BOLTZ_PARTITION = SLURM_CFG.get("boltz", {}).get("partition", SLURM_CFG.get("partition", ""))
BOLTZ_ACCOUNT   = SLURM_CFG.get("account", "")


def boltz_chunk_input(wildcards):
    """Input for chunk_yamls_for_boltz: depend on MSA completion if MSA is enabled."""
    inputs = {"yaml_dir": YAML_SOURCE_DIR}
    if RUN_MSA:
        inputs["msa_done"] = f"{OUTPUT}/.msa_complete"
    return inputs


checkpoint chunk_yamls_for_boltz:
    """Split YAML files into chunk directories for parallel boltz predict."""
    input:
        unpack(boltz_chunk_input),
    output:
        manifest = f"{BOLTZ_CHUNKS}/manifest.txt",
    params:
        max_files = BOLTZ_CFG.get("max_files_per_job", 25),
    resources:
        cpus_per_task = 1,
        mem_mb = 2000,
        runtime = 10,
        slurm_partition = BOLTZ_PARTITION,
        slurm_account = BOLTZ_ACCOUNT,
    shell:
        """
        python workflow/scripts/prepare_boltz_chunks.py \
            --yaml_dir {input.yaml_dir} \
            --output_dir {BOLTZ_CHUNKS} \
            --max_files_per_job {params.max_files}
        """


def get_boltz_chunk_ids(wildcards):
    """Return chunk IDs from the boltz manifest after checkpoint."""
    manifest = checkpoints.chunk_yamls_for_boltz.get().output.manifest
    with open(manifest) as f:
        dirs = [line.strip() for line in f if line.strip()]
    ids = []
    for d in dirs:
        name = d.rstrip("/").split("/")[-1]
        ids.append(name.replace("chunk_", ""))
    return ids


rule run_boltz_predict:
    """Run boltz predict on a chunk directory of YAML files."""
    input:
        chunk_dir = f"{BOLTZ_CHUNKS}/chunk_{{chunk_id}}",
    output:
        done = f"{BOLTZ_CHUNKS}/chunk_{{chunk_id}}_output/.done",
    params:
        output_dir = f"{BOLTZ_CHUNKS}/chunk_{{chunk_id}}_output",
        cache_dir = BOLTZ_CFG.get("cache_dir", ""),
        env_path = BOLTZ_CFG.get("env_path", ""),
        recycling_steps = BOLTZ_CFG.get("recycling_steps", 10),
        diffusion_samples = BOLTZ_CFG.get("diffusion_samples", 25),
        container_cmd = container_cmd("boltz"),
    resources:
        cpus_per_task = 8,
        mem_mb = 16000,
        runtime = 60,
        slurm_partition = BOLTZ_PARTITION,
        slurm_account = BOLTZ_ACCOUNT,
        slurm_extra = "'--gpus-per-node=1'",
    shell:
        """
        set -euo pipefail
        export CUDA_VISIBLE_DEVICES=0
        export NUM_GPU_DEVICES=1
        export TMPDIR="${{SLURM_TMPDIR:-/tmp}}"
        export TRITON_CACHE_DIR="${{TMPDIR}}/triton_cache_$$"
        mkdir -p "$TRITON_CACHE_DIR"
        mkdir -p {params.output_dir}

        if [ -n "{params.container_cmd}" ]; then
            {params.container_cmd} \
                --env TRITON_CACHE_DIR=$TRITON_CACHE_DIR \
                boltz predict {input.chunk_dir} \
                    --cache {params.cache_dir} --out_dir {params.output_dir} \
                    --devices 1 --accelerator gpu \
                    --recycling_steps {params.recycling_steps} \
                    --diffusion_samples {params.diffusion_samples} --override
        else
            module load python/3.12.8-fasrc01 gcc/14.2.0-fasrc01 cuda/12.9.1-fasrc01 cudnn/9.10.2.21_cuda12-fasrc01 || true
            mamba activate {params.env_path}
            boltz predict {input.chunk_dir} \
                --cache {params.cache_dir} --out_dir {params.output_dir} \
                --devices 1 --accelerator gpu \
                --recycling_steps {params.recycling_steps} \
                --diffusion_samples {params.diffusion_samples} --override
        fi

        touch {output.done}
        """


rule organize_boltz_chunk:
    """Pick highest-confidence model per sequence and copy to sequences/{seq}/boltz/."""
    input:
        done = f"{BOLTZ_CHUNKS}/chunk_{{chunk_id}}_output/.done",
    output:
        organized = f"{BOLTZ_CHUNKS}/chunk_{{chunk_id}}_output/.organized",
    params:
        boltz_output_dir = f"{BOLTZ_CHUNKS}/chunk_{{chunk_id}}_output",
        sequences_dir = SEQUENCES_DIR,
        delete_msa = "--delete_msa" if BOLTZ_CFG.get("delete_msa_after_processing", False) else "",
    resources:
        cpus_per_task = 4,
        mem_mb = 8000,
        runtime = 60,
        slurm_partition = BOLTZ_PARTITION,
        slurm_account = BOLTZ_ACCOUNT,
    shell:
        """
        python workflow/scripts/organize_boltz_outputs.py \
            --boltz_output_dir {params.boltz_output_dir} \
            --chunk_id {wildcards.chunk_id} \
            --sequences_dir {params.sequences_dir} \
            {params.delete_msa}

        touch {output.organized}
        """


def aggregate_boltz_organized(wildcards):
    """Collect all .organized sentinels after checkpoint expansion."""
    chunk_ids = get_boltz_chunk_ids(wildcards)
    return expand(f"{BOLTZ_CHUNKS}/chunk_{{chunk_id}}_output/.organized", chunk_id=chunk_ids)


rule boltz_complete:
    """Aggregate sentinel: all Boltz chunks organized."""
    input:
        aggregate_boltz_organized,
    output:
        done = f"{OUTPUT}/.boltz_complete",
    localrule: True
    shell:
        "touch {output.done}"
