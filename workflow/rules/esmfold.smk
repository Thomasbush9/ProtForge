"""
ESMFold Stage Rules
===================
checkpoint chunk_yamls_for_esmfold -> run_esmfold_chunk (per chunk) -> esmfold_complete (aggregate)

Calls slurm_scripts/run_esmfold.py to fold sequences via facebook/esmfold_v1.
Reuses workflow/scripts/chunk_yamls_for_esm.py for chunking (format-agnostic).
"""

ESMFOLD_CFG    = config.get("esmfold", {})
ESMFOLD_CHUNKS = f"{OUTPUT}/esmfold_chunks"

# input_type: "yaml" (post-MSA, default) | "fasta" (raw .fasta dir, independent of MSA/Boltz)
ESMFOLD_INPUT_TYPE = ESMFOLD_CFG.get("input_type", "yaml").lower()

if ESMFOLD_INPUT_TYPE == "fasta":
    ESMFOLD_SOURCE_DIR = config["input"].get("fasta_dir", "")
    ESMFOLD_GLOB = "*.fasta"
else:
    _yaml_dir_override_ef = config["input"].get("yaml_dir", "")
    ESMFOLD_SOURCE_DIR = _yaml_dir_override_ef if _yaml_dir_override_ef else SEQUENCES_DIR
    ESMFOLD_GLOB = "*.yaml"

ESMFOLD_PARTITION = SLURM_CFG.get("esmfold", {}).get("partition", SLURM_CFG.get("partition", ""))
ESMFOLD_ACCOUNT   = SLURM_CFG.get("account", "")


def esmfold_chunk_input(wildcards):
    """Input for chunk_yamls_for_esmfold: depend on upstream completion only in yaml mode."""
    inputs = {}
    if ESMFOLD_INPUT_TYPE == "fasta":
        return inputs  # raw fasta is independent of MSA/Boltz
    if RUN_BOLTZ:
        inputs["upstream_done"] = f"{OUTPUT}/.boltz_complete"
    elif RUN_MSA:
        inputs["upstream_done"] = f"{OUTPUT}/.msa_complete"
    return inputs


checkpoint chunk_yamls_for_esmfold:
    """Split input paths into N chunk files for parallel ESMFold processing."""
    input:
        unpack(esmfold_chunk_input),
    output:
        manifest = f"{ESMFOLD_CHUNKS}/manifest.txt",
    params:
        source_dir = ESMFOLD_SOURCE_DIR,
        glob_pattern = ESMFOLD_GLOB,
        num_chunks = ESMFOLD_CFG.get("num_chunks", 1),
        esmfold_chunks_dir = ESMFOLD_CHUNKS,
    localrule: True
    shell:
        """
        python workflow/scripts/chunk_yamls_for_esm.py \
            --yaml_dir {params.source_dir} \
            --output_dir {params.esmfold_chunks_dir} \
            --num_chunks {params.num_chunks} \
            --pattern '{params.glob_pattern}'
        """


def get_esmfold_chunk_ids(wildcards):
    """Return chunk IDs from the ESMFold manifest after checkpoint."""
    manifest = checkpoints.chunk_yamls_for_esmfold.get().output.manifest
    with open(manifest) as f:
        lines = [line.strip() for line in f if line.strip()]
    ids = []
    for path_str in lines:
        name = path_str.rstrip("/").split("/")[-1]
        ids.append(name.replace("id_", "").replace(".txt", ""))
    return ids


rule run_esmfold_chunk:
    """Fold sequences in a chunk with facebook/esmfold_v1."""
    input:
        fasta_list = f"{ESMFOLD_CHUNKS}/id_{{chunk_id}}.txt",
        script = "slurm_scripts/run_esmfold.py",
    output:
        done = f"{ESMFOLD_CHUNKS}/chunk_{{chunk_id}}.done",
    benchmark:
        f"{OUTPUT}/benchmarks/esmfold/esmfold_chunk_{{chunk_id}}.tsv"
    log:
        f"{OUTPUT}/logs/esmfold/esmfold_chunk_{{chunk_id}}.log",
    params:
        output_dir = SEQUENCES_DIR,
        env_path = ESMFOLD_CFG.get("env_path", ""),
        cache_dir = ESMFOLD_CFG.get("cache_dir", ""),
        container_cmd = container_cmd("esmfold"),
        esmfold_chunks_dir = ESMFOLD_CHUNKS,
    resources:
        cpus_per_task = stage_resource("esmfold", "cpus_per_task", 8),
        mem_mb        = stage_resource("esmfold", "mem_mb", 32000),
        runtime       = stage_resource("esmfold", "runtime", 120),
        slurm_partition = ESMFOLD_PARTITION,
        slurm_account = ESMFOLD_ACCOUNT,
        slurm_extra = slurm_extra(gpu=stage_uses_gpu("esmfold", True)),
    shell:
        """
        set -euo pipefail
        exec > {log} 2>&1
        set -x

        export PYTHONPATH="$(pwd)${{PYTHONPATH:+:$PYTHONPATH}}"

        if [ -n "{params.container_cmd}" ]; then
            {params.container_cmd} \
                --env HF_HOME={params.cache_dir} \
                python /opt/protforge/run_esmfold.py \
                    --fasta_list {input.fasta_list} \
                    --output_dir {params.output_dir} \
                    --processed_paths_file {params.esmfold_chunks_dir}/processed_paths_{wildcards.chunk_id}.txt \
                    --cache_dir {params.cache_dir}
        else
            module load python gcc/14.2.0-fasrc01 cuda/12.9.1-fasrc01 cudnn/9.10.2.21_cuda12-fasrc01 || true

            set +u
            source "$(conda info --base)/etc/profile.d/mamba.sh"
            mamba activate "{params.env_path}"
            set -u

            python {input.script} \
                --fasta_list {input.fasta_list} \
                --output_dir {params.output_dir} \
                --processed_paths_file {params.esmfold_chunks_dir}/processed_paths_{wildcards.chunk_id}.txt \
                --cache_dir {params.cache_dir}
        fi

        touch {output.done}
        """


def aggregate_esmfold_done(wildcards):
    """Collect all .done sentinels after checkpoint expansion."""
    chunk_ids = get_esmfold_chunk_ids(wildcards)
    return expand(f"{ESMFOLD_CHUNKS}/chunk_{{chunk_id}}.done", chunk_id=chunk_ids)


rule esmfold_complete:
    """Aggregate sentinel: all ESMFold chunks processed."""
    input:
        aggregate_esmfold_done,
    output:
        done = f"{OUTPUT}/.esmfold_complete",
    localrule: True
    shell:
        "touch {output.done}"
