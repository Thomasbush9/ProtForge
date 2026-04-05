"""
ESM Stage Rules
================
checkpoint chunk_yamls_for_esm -> run_esm_chunk (per chunk) -> esm_complete (aggregate)

Calls the existing slurm_scripts/run_esm.py directly.
"""

ESM_CFG      = config.get("esm", {})
ESM_CHUNKS   = f"{OUTPUT}/esm_chunks"

# YAML source: user-provided yaml_dir, or auto-discover from {OUTPUT}/sequences/
_yaml_dir_override = config["input"].get("yaml_dir", "")
ESM_YAML_SOURCE = _yaml_dir_override if _yaml_dir_override else SEQUENCES_DIR

ESM_PARTITION = SLURM_CFG.get("esm", {}).get("partition", SLURM_CFG.get("partition", ""))
ESM_ACCOUNT   = SLURM_CFG.get("account", "")


def esm_chunk_input(wildcards):
    """Input for chunk_yamls_for_esm: depend on upstream completion if enabled."""
    inputs = {}
    if RUN_BOLTZ:
        inputs["upstream_done"] = f"{OUTPUT}/.boltz_complete"
    elif RUN_MSA:
        inputs["upstream_done"] = f"{OUTPUT}/.msa_complete"
    return inputs


checkpoint chunk_yamls_for_esm:
    """Split YAML paths into N chunk files for parallel ESM processing."""
    input:
        unpack(esm_chunk_input),
    output:
        manifest = f"{ESM_CHUNKS}/manifest.txt",
    params:
        yaml_dir = ESM_YAML_SOURCE,
        num_chunks = ESM_CFG.get("num_chunks", 1),
        esm_chunks_dir = ESM_CHUNKS,
    localrule: True
    shell:
        """
        python workflow/scripts/chunk_yamls_for_esm.py \
            --yaml_dir {params.yaml_dir} \
            --output_dir {params.esm_chunks_dir} \
            --num_chunks {params.num_chunks}
        """


def get_esm_chunk_ids(wildcards):
    """Return chunk IDs from the ESM manifest after checkpoint."""
    manifest = checkpoints.chunk_yamls_for_esm.get().output.manifest
    with open(manifest) as f:
        lines = [line.strip() for line in f if line.strip()]
    ids = []
    for path_str in lines:
        # id_0.txt -> 0
        name = path_str.rstrip("/").split("/")[-1]
        ids.append(name.replace("id_", "").replace(".txt", ""))
    return ids


rule run_esm_chunk:
    """Run ESM embeddings/logits on a chunk of YAML files."""
    input:
        fasta_list = f"{ESM_CHUNKS}/id_{{chunk_id}}.txt",
        script = "slurm_scripts/run_esm.py",
    output:
        done = f"{ESM_CHUNKS}/chunk_{{chunk_id}}.done",
    benchmark:
        f"{OUTPUT}/benchmarks/esm/esm_chunk_{{chunk_id}}.tsv"
    log:
        f"{ESM_CHUNKS}/esm_chunk_{{chunk_id}}.log",
    params:
        output_dir = SEQUENCES_DIR,
        env_path = ESM_CFG.get("env_path", ""),
        cache_dir = ESM_CFG.get("cache_dir", ""),
        container_cmd = container_cmd("esm"),
        esm_chunks_dir = ESM_CHUNKS,
    resources:
        cpus_per_task = 16,
        mem_mb = 32000,
        runtime = 60,
        slurm_partition = ESM_PARTITION,
        slurm_account = ESM_ACCOUNT,
        slurm_extra = slurm_extra(gpu=True),
    shell:
        """
        set -euo pipefail
        exec > {log} 2>&1
        set -x

        # Add ProtForge root to PYTHONPATH (for utils.utils import)
        export PYTHONPATH="$(pwd)${{PYTHONPATH:+:$PYTHONPATH}}"

        if [ -n "{params.container_cmd}" ]; then
            {params.container_cmd} \
                --env TORCH_HOME={params.cache_dir} \
                --env HF_HOME={params.cache_dir} \
                python /opt/protforge/run_esm.py \
                    --fasta_list {input.fasta_list} \
                    --output_dir {params.output_dir} \
                    --processed_paths_file {params.esm_chunks_dir}/processed_paths_{wildcards.chunk_id}.txt
        else
            module load gcc/14.2.0-fasrc01 cuda/12.9.1-fasrc01 cudnn/9.10.2.21_cuda12-fasrc01 || true
            if [ -n "{params.cache_dir}" ]; then
                export TORCH_HOME="{params.cache_dir}"
                export HF_HOME="{params.cache_dir}"
            fi
            # Use conda env's Python directly (avoids activation conflicts with Snakemake)
            {params.env_path}/bin/python {input.script} \
                --fasta_list {input.fasta_list} \
                --output_dir {params.output_dir} \
                --processed_paths_file {params.esm_chunks_dir}/processed_paths_{wildcards.chunk_id}.txt
        fi

        touch {output.done}
        """


def aggregate_esm_done(wildcards):
    """Collect all .done sentinels after checkpoint expansion."""
    chunk_ids = get_esm_chunk_ids(wildcards)
    return expand(f"{ESM_CHUNKS}/chunk_{{chunk_id}}.done", chunk_id=chunk_ids)


rule esm_complete:
    """Aggregate sentinel: all ESM chunks processed."""
    input:
        aggregate_esm_done,
    output:
        done = f"{OUTPUT}/.esm_complete",
    localrule: True
    shell:
        "touch {output.done}"
