"""
Boltz Stage Rules
==================
checkpoint chunk_yamls_for_boltz -> run_boltz_predict (per chunk × run) ->
organize_boltz_chunk (per chunk × run) -> boltz_complete (aggregate)

Supports multiple independent runs per sequence via boltz.num_runs.
When num_runs > 1, outputs are organized into:
  sequences/{seq}/boltz/run_0/, sequences/{seq}/boltz/run_1/, ...
When num_runs == 1, outputs go directly to sequences/{seq}/boltz/ (legacy).
"""

BOLTZ_CFG     = config.get("boltz", {})
BOLTZ_CHUNKS  = f"{OUTPUT}/boltz_chunks"
NUM_RUNS      = BOLTZ_CFG.get("num_runs", 1)

# Determine YAML source: from MSA stage (sequences/) or user-provided yaml_dir
YAML_SOURCE_DIR = f"{OUTPUT}/sequences" if RUN_MSA else config["input"].get("yaml_dir", "")

BOLTZ_PARTITION = SLURM_CFG.get("boltz", {}).get("partition", SLURM_CFG.get("partition", ""))
BOLTZ_ACCOUNT   = SLURM_CFG.get("account", "")

# Optional opt-in CLI flags from boltz.advanced (set via the webapp dialog).
# Anything missing here falls back to Boltz's own defaults.
_BOLTZ_FLAG_OPTIONS = {
    "affinity_mw_correction", "subsample_msa", "no_kernels",
    "use_potentials", "write_full_pae", "write_full_pde",
}
# Map yaml-friendly key -> actual CLI flag name (only odd ones)
_BOLTZ_CLI_NAME = {"preprocessing_threads": "preprocessing-threads"}


def _build_boltz_extra_args(cfg: dict) -> str:
    """Convert boltz.advanced dict into a `boltz predict` flag string."""
    adv = cfg.get("advanced", {}) or {}
    parts = []
    for key, value in adv.items():
        cli = _BOLTZ_CLI_NAME.get(key, key)
        if key in _BOLTZ_FLAG_OPTIONS:
            if value:
                parts.append(f"--{cli}")
        else:
            if value is None or value == "":
                continue
            parts.append(f"--{cli} {value}")
    return " ".join(parts)


BOLTZ_EXTRA_ARGS = _build_boltz_extra_args(BOLTZ_CFG)


def boltz_chunk_input(wildcards):
    """Input for chunk_yamls_for_boltz: depend on MSA completion if MSA is enabled."""
    inputs = {}
    if RUN_MSA:
        inputs["msa_done"] = f"{OUTPUT}/.msa_complete"
    else:
        inputs["yaml_dir"] = YAML_SOURCE_DIR
    return inputs


def _boltz_chunker_extra() -> str:
    parts = []
    max_seq_len = BOLTZ_CFG.get("max_seq_len")
    if max_seq_len is not None:
        parts.append(f"--max_seq_len {int(max_seq_len)}")
    bin_args = binning_args(BOLTZ_CFG)
    if bin_args:
        parts.append(bin_args)
    return " ".join(parts)


BOLTZ_CHUNKER_EXTRA = _boltz_chunker_extra()
BOLTZ_CHUNKS_TSV = f"{BOLTZ_CHUNKS}/chunks.tsv"


checkpoint chunk_yamls_for_boltz:
    """Split YAML files into chunk directories for parallel boltz predict."""
    input:
        unpack(boltz_chunk_input),
    output:
        manifest = f"{BOLTZ_CHUNKS}/manifest.txt",
    params:
        yaml_dir = YAML_SOURCE_DIR,
        max_files = BOLTZ_CFG.get("max_files_per_job", 25),
        chunker_extra = BOLTZ_CHUNKER_EXTRA,
    localrule: True
    shell:
        """
        python workflow/scripts/prepare_boltz_chunks.py \
            --yaml_dir {params.yaml_dir} \
            --output_dir {BOLTZ_CHUNKS} \
            --max_files_per_job {params.max_files} \
            {params.chunker_extra}
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
        done = f"{BOLTZ_CHUNKS}/chunk_{{chunk_id}}_run_{{run_id}}_output/.done",
    benchmark:
        f"{OUTPUT}/benchmarks/boltz/predict_{{chunk_id}}_run_{{run_id}}.tsv"
    log:
        f"{OUTPUT}/logs/boltz/predict_{{chunk_id}}_run_{{run_id}}.log"
    params:
        output_dir = f"{BOLTZ_CHUNKS}/chunk_{{chunk_id}}_run_{{run_id}}_output",
        cache_dir = BOLTZ_CFG.get("cache_dir", ""),
        env_path = BOLTZ_CFG.get("env_path", ""),
        recycling_steps = BOLTZ_CFG.get("recycling_steps", 10),
        diffusion_samples = BOLTZ_CFG.get("diffusion_samples", 25),
        extra_args = BOLTZ_EXTRA_ARGS,
        container_cmd = container_cmd("boltz"),
    resources:
        cpus_per_task = stage_resource("boltz", "cpus_per_task", 8),
        mem_mb        = lambda wc: chunk_resource(
            BOLTZ_CHUNKS_TSV, wc.chunk_id, "mem_mb",
            stage_resource("boltz", "mem_mb", 16000),
        ),
        runtime       = lambda wc: chunk_resource(
            BOLTZ_CHUNKS_TSV, wc.chunk_id, "runtime_min",
            stage_resource("boltz", "runtime", 60),
        ),
        slurm_partition = BOLTZ_PARTITION,
        slurm_account = BOLTZ_ACCOUNT,
        slurm_extra = slurm_extra(gpu=stage_uses_gpu("boltz", True)),
    shell:
        """
        set -euo pipefail
        exec > {log} 2>&1
        set -x

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
                    --diffusion_samples {params.diffusion_samples} \
                    {params.extra_args} --override
        else
            module load python/3.12.8-fasrc01 gcc/14.2.0-fasrc01 cuda/12.9.1-fasrc01 cudnn/9.10.2.21_cuda12-fasrc01 || true
            mamba activate {params.env_path}
            boltz predict {input.chunk_dir} \
                --cache {params.cache_dir} --out_dir {params.output_dir} \
                --devices 1 --accelerator gpu \
                --recycling_steps {params.recycling_steps} \
                --diffusion_samples {params.diffusion_samples} \
                {params.extra_args} --override
        fi

        touch {output.done}
        """


rule organize_boltz_chunk:
    """Copy model outputs to sequences/{seq}/boltz/ then delete raw chunk output."""
    input:
        done = f"{BOLTZ_CHUNKS}/chunk_{{chunk_id}}_run_{{run_id}}_output/.done",
    output:
        organized = f"{BOLTZ_CHUNKS}/chunk_{{chunk_id}}_run_{{run_id}}.organized",
    params:
        boltz_output_dir = f"{BOLTZ_CHUNKS}/chunk_{{chunk_id}}_run_{{run_id}}_output",
        sequences_dir = SEQUENCES_DIR,
        delete_msa = "--delete_msa" if BOLTZ_CFG.get("delete_msa_after_processing", False) else "",
        subdirectory = lambda wc: f"--subdirectory run_{wc.run_id}" if NUM_RUNS > 1 else "",
    localrule: True
    shell:
        """
        python workflow/scripts/organize_boltz_outputs.py \
            --boltz_output_dir {params.boltz_output_dir} \
            --chunk_id {wildcards.chunk_id} \
            --sequences_dir {params.sequences_dir} \
            {params.delete_msa} {params.subdirectory}

        rm -rf {params.boltz_output_dir}

        touch {output.organized}
        """


def aggregate_boltz_organized(wildcards):
    """Collect all .organized sentinels across chunks × runs."""
    chunk_ids = get_boltz_chunk_ids(wildcards)
    run_ids = list(range(NUM_RUNS))
    return expand(
        f"{BOLTZ_CHUNKS}/chunk_{{chunk_id}}_run_{{run_id}}.organized",
        chunk_id=chunk_ids, run_id=run_ids,
    )


rule boltz_complete:
    """Aggregate sentinel: all Boltz chunks × runs organized."""
    input:
        aggregate_boltz_organized,
    output:
        done = f"{OUTPUT}/.boltz_complete",
    localrule: True
    shell:
        "touch {output.done}"
