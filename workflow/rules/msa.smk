"""
MSA Stage Rules
================
checkpoint chunk_fastas -> run_colabfold_search (per chunk) ->
scatter_msa_and_create_yaml (per chunk) -> msa_complete (aggregate)
"""

MSA_CFG       = config.get("msa", {})
FASTA_DIR     = config["input"]["fasta_dir"]
MSA_CHUNKS    = f"{OUTPUT}/msa_chunks"

# Per-stage SLURM overrides
MSA_PARTITION = SLURM_CFG.get("msa", {}).get("partition", SLURM_CFG.get("partition", ""))
MSA_ACCOUNT   = SLURM_CFG.get("account", "")


checkpoint chunk_fastas:
    """Split input FASTAs into chunk directories with combined.fasta + file_list.txt."""
    input:
        fasta_dir = FASTA_DIR,
    output:
        manifest = f"{MSA_CHUNKS}/manifest.txt",
    params:
        max_files = MSA_CFG.get("max_files_per_job", 25),
    localrule: True
    shell:
        """
        python workflow/scripts/chunk_fastas.py \
            --input_dir {input.fasta_dir} \
            --output_dir {MSA_CHUNKS} \
            --max_files_per_job {params.max_files}
        """


def get_msa_chunk_dirs(wildcards):
    """After chunk_fastas checkpoint, read manifest to get chunk directories."""
    manifest = checkpoints.chunk_fastas.get().output.manifest
    with open(manifest) as f:
        return [line.strip() for line in f if line.strip()]


def get_msa_chunk_ids(wildcards):
    """Return chunk IDs (0, 1, 2, ...) from the manifest."""
    dirs = get_msa_chunk_dirs(wildcards)
    # Extract chunk_N -> N
    ids = []
    for d in dirs:
        name = d.rstrip("/").split("/")[-1]  # chunk_0, chunk_1, ...
        ids.append(name.replace("chunk_", ""))
    return ids


rule run_colabfold_search:
    """Run colabfold_search on a combined FASTA for one chunk."""
    input:
        combined = f"{MSA_CHUNKS}/chunk_{{chunk_id}}/combined.fasta",
    output:
        done = f"{MSA_CHUNKS}/chunk_{{chunk_id}}/colabfold_output/.done",
    benchmark:
        f"{OUTPUT}/benchmarks/msa/colabfold_search_{{chunk_id}}.tsv"
    params:
        output_dir = f"{MSA_CHUNKS}/chunk_{{chunk_id}}/colabfold_output",
        mmseq2_db = MSA_CFG.get("mmseq2_db", ""),
        colabfold_bin = MSA_CFG.get("colabfold_bin", ""),
        threads = 4,
        container_cmd = container_cmd("colabfold"),
    resources:
        cpus_per_task = 4,
        mem_mb = 48000,
        runtime = 60,
        slurm_partition = MSA_PARTITION,
        slurm_account = MSA_ACCOUNT,
        slurm_extra = slurm_extra(gpu=True),
    shell:
        """
        set -euo pipefail
        export CUDA_VISIBLE_DEVICES=0
        export NUM_GPU_DEVICES=1

        # Clean stale temp files from previous failed runs (tmp1, tmp2, tmp3, ...)
        # but keep "tmp" itself — colabfold expects to rmtree("tmp") at the end
        rm -rf {params.output_dir}/tmp[0-9]* {params.output_dir}/.done
        mkdir -p {params.output_dir}/tmp

        if [ -n "{params.container_cmd}" ]; then
            {params.container_cmd} \
                colabfold_search {input.combined} {params.mmseq2_db} {params.output_dir} \
                    --thread {params.threads} --gpu 1
        else
            module load python/3.12.8-fasrc01 gcc/14.2.0-fasrc01 cuda/12.9.1-fasrc01 cudnn/9.10.2.21_cuda12-fasrc01 || true
            if [ -n "{params.colabfold_bin}" ]; then
                export PATH="{params.colabfold_bin}:$PATH"
            fi
            colabfold_search {input.combined} {params.mmseq2_db} {params.output_dir} \
                --thread {params.threads} --gpu 1
        fi

        touch {output.done}
        """


rule scatter_msa_and_create_yaml:
    """Scatter a3m files to per-sequence dirs and create Boltz YAML files."""
    input:
        done = f"{MSA_CHUNKS}/chunk_{{chunk_id}}/colabfold_output/.done",
        file_list = f"{MSA_CHUNKS}/chunk_{{chunk_id}}/file_list.txt",
    output:
        scatter_done = f"{MSA_CHUNKS}/chunk_{{chunk_id}}/.scatter_done",
    params:
        colabfold_output = f"{MSA_CHUNKS}/chunk_{{chunk_id}}/colabfold_output",
        sequences_dir = SEQUENCES_DIR,
    localrule: True
    shell:
        """
        python workflow/scripts/organize_msa_outputs.py \
            --file_list {input.file_list} \
            --colabfold_output {params.colabfold_output} \
            --sequences_dir {params.sequences_dir}

        touch {output.scatter_done}
        """


def aggregate_msa_scatter_done(wildcards):
    """Collect all .scatter_done sentinels after checkpoint expansion."""
    chunk_ids = get_msa_chunk_ids(wildcards)
    return expand(f"{MSA_CHUNKS}/chunk_{{chunk_id}}/.scatter_done", chunk_id=chunk_ids)


rule msa_complete:
    """Aggregate sentinel: all MSA chunks have been scattered and YAMLs created."""
    input:
        aggregate_msa_scatter_done,
    output:
        done = f"{OUTPUT}/.msa_complete",
    localrule: True
    shell:
        "touch {output.done}"
