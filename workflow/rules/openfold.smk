"""
OpenFold3 Stage Rules
=====================
checkpoint chunk_yamls_for_openfold -> run_openfold_predict (per chunk) ->
organize_openfold_chunk (per chunk) -> openfold_complete (aggregate)

Each chunk is one OpenFold query JSON = one SLURM job. A job folds all of its
queries, distributing them across `openfold.gpus_per_job` GPUs on a single node
via PyTorch Lightning (pl_trainer_args.devices). num jobs == num JSONs.

Runs off the MSA-stage YAMLs (in parallel with Boltz). The MSA a3m is copied to
the OpenFold-expected uniref90_hits.a3m at chunk time; the container reads it via
each query's main_msa_file_paths (--use-msa-server=False).
Outputs -> sequences/{seq}/openfold/.
"""

import json as _json

OPENFOLD_CFG    = config.get("openfold", {})
OPENFOLD_CHUNKS = f"{OUTPUT}/openfold_chunks"

# YAML source: MSA stage (sequences/) or user-provided yaml_dir (MSA skipped).
OF_YAML_SOURCE_DIR = f"{OUTPUT}/sequences" if RUN_MSA else config["input"].get("yaml_dir", "")

OPENFOLD_PARTITION = SLURM_CFG.get("openfold", {}).get("partition", SLURM_CFG.get("partition", ""))
OPENFOLD_ACCOUNT   = SLURM_CFG.get("account", "")

# GPUs per job (clamped 1..4): drives pl_trainer_args.devices + --gpus-per-node.
OF_GPUS_PER_JOB = max(1, min(4, int(OPENFOLD_CFG.get("gpus_per_job", 1))))

# When MSA is skipped the user's yaml_dir (and the a3m paths inside it) differs
# from sequences/ and must be bound too. Same pattern as boltz.smk.
_OF_EXTRA_YAML_BIND = (
    f"-B {OF_YAML_SOURCE_DIR}:{OF_YAML_SOURCE_DIR}"
    if OF_YAML_SOURCE_DIR and OF_YAML_SOURCE_DIR != SEQUENCES_DIR
    else ""
)


def _openfold_chunker_extra() -> str:
    """CLI flags for prepare_openfold_chunks.py from config."""
    parts = []
    num_batches = OPENFOLD_CFG.get("num_batches")
    if num_batches:
        parts.append(f"--num_batches {int(num_batches)}")
    parts.append(f"--max_files_per_job {int(OPENFOLD_CFG.get('max_files_per_job', 25))}")
    parts.append(f"--devices {OF_GPUS_PER_JOB}")
    parts.append(f"--num_workers {int(OPENFOLD_CFG.get('num_workers', 10))}")
    parts.append(f"--structure_format {OPENFOLD_CFG.get('structure_format', 'cif')}")
    if OPENFOLD_CFG.get("write_full_confidence", True):
        parts.append("--write_full_confidence")
    seeds = OPENFOLD_CFG.get("seeds") or []
    if seeds:
        parts.append("--seeds " + ",".join(str(int(s)) for s in seeds))
    recycling = OPENFOLD_CFG.get("recycling_iters")
    if recycling is not None:
        parts.append(f"--recycling_iters {int(recycling)}")
    advanced = OPENFOLD_CFG.get("advanced") or {}
    if advanced:
        parts.append("--runner_advanced_json " + _shlex.quote(_json.dumps(advanced)))
    return " ".join(parts)


OPENFOLD_CHUNKER_EXTRA = _openfold_chunker_extra()


def openfold_chunk_input(wildcards):
    """Depend on MSA completion when MSA is enabled, else on the yaml_dir."""
    if RUN_MSA:
        return {"msa_done": f"{OUTPUT}/.msa_complete"}
    return {"yaml_dir": OF_YAML_SOURCE_DIR}


checkpoint chunk_yamls_for_openfold:
    """Split YAMLs into OpenFold query JSONs (one per job) + a shared runner.yaml."""
    input:
        unpack(openfold_chunk_input),
    output:
        manifest = f"{OPENFOLD_CHUNKS}/manifest.txt",
    params:
        yaml_dir = OF_YAML_SOURCE_DIR,
        chunker_extra = OPENFOLD_CHUNKER_EXTRA,
    localrule: True
    shell:
        """
        python workflow/scripts/prepare_openfold_chunks.py \
            --yaml_dir {params.yaml_dir} \
            --output_dir {OPENFOLD_CHUNKS} \
            {params.chunker_extra}
        """


def get_openfold_chunk_ids(wildcards):
    """Return chunk IDs from the manifest (paths to chunk_{i}.json)."""
    manifest = checkpoints.chunk_yamls_for_openfold.get().output.manifest
    ids = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name = line.rstrip("/").split("/")[-1]      # chunk_{i}.json
            ids.append(name.removeprefix("chunk_").removesuffix(".json"))
    return ids


rule run_openfold_predict:
    """Fold one chunk JSON's queries (batched across GPUs) inside the OpenFold image."""
    input:
        chunk_json = f"{OPENFOLD_CHUNKS}/chunk_{{chunk_id}}.json",
        runner     = f"{OPENFOLD_CHUNKS}/runner.yaml",
    output:
        done = f"{OPENFOLD_CHUNKS}/chunk_{{chunk_id}}_output/.done",
    benchmark:
        f"{OUTPUT}/benchmarks/openfold/predict_{{chunk_id}}.tsv"
    log:
        f"{OUTPUT}/logs/openfold/predict_{{chunk_id}}.log"
    params:
        output_dir = f"{OPENFOLD_CHUNKS}/chunk_{{chunk_id}}_output",
        chunks_dir = OPENFOLD_CHUNKS,
        cache_dir = OPENFOLD_CFG.get("cache_dir", ""),
        msa_dir = SEQUENCES_DIR,
        extra_yaml_bind = _OF_EXTRA_YAML_BIND,
        num_diffusion_samples = OPENFOLD_CFG.get("num_diffusion_samples", 5),
        num_model_seeds = OPENFOLD_CFG.get("num_model_seeds", 1),
        runtime = CONTAINER_RUNTIME,
        sif = container_sif("openfold"),
    resources:
        cpus_per_task = stage_resource("openfold", "cpus_per_task", 8),
        mem_mb        = stage_resource("openfold", "mem_mb", 48000),
        runtime       = stage_resource("openfold", "runtime", 60),
        openfold_jobs = 1,
        slurm_partition = OPENFOLD_PARTITION,
        slurm_account = OPENFOLD_ACCOUNT,
        slurm_extra = slurm_extra(gpu=stage_uses_gpu("openfold", True),
                                  gpu_count=OF_GPUS_PER_JOB),
    shell:
        """
        set -euo pipefail
        exec > {log} 2>&1
        set -x

        export TMPDIR="${{SLURM_TMPDIR:-/tmp}}"
        export TRITON_CACHE_DIR="${{TMPDIR}}/triton_cache_$$"
        mkdir -p "$TRITON_CACHE_DIR" {params.output_dir}

        if [ -z "{params.sif}" ]; then
            echo "ERROR: no OpenFold container configured. Set containers.openfold" \
                 "(or containers.gpu) in config to the openfold .sif path." >&2
            exit 1
        fi
        if [ -z "{params.cache_dir}" ]; then
            echo "ERROR: openfold.cache_dir is unset. Point it at the OpenFold" \
                 "weights/CCD cache (bound to /models/openfold)." >&2
            exit 1
        fi

        # Same-path binds so JSON main_msa_file_paths (under sequences/) and the
        # chunk JSON + runner.yaml resolve identically in-container. Weights cache
        # at /models/openfold matches OPENFOLD_CACHE (mirrors batch_openfold3_test.sh).
        {params.runtime} exec --nv --cleanenv \
            --env OPENFOLD_CACHE=/models/openfold \
            --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
            -B "${{SLURM_TMPDIR:-/tmp}}":/tmp \
            -B {params.cache_dir}:/models/openfold:ro \
            -B {params.chunks_dir}:{params.chunks_dir} \
            -B {params.msa_dir}:{params.msa_dir} \
            {params.extra_yaml_bind} \
            -B {params.output_dir}:{params.output_dir} \
            {params.sif} \
            run_openfold predict \
                --query-json {input.chunk_json} \
                --output-dir {params.output_dir} \
                --use-msa-server=False \
                --num-diffusion-samples {params.num_diffusion_samples} \
                --num-model-seeds {params.num_model_seeds} \
                --runner-yaml {input.runner}

        touch {output.done}
        """


rule organize_openfold_chunk:
    """Copy top-N samples per query to sequences/{seq}/openfold/, drop raw output."""
    input:
        done = f"{OPENFOLD_CHUNKS}/chunk_{{chunk_id}}_output/.done",
    output:
        organized = f"{OPENFOLD_CHUNKS}/chunk_{{chunk_id}}.organized",
    params:
        openfold_output_dir = f"{OPENFOLD_CHUNKS}/chunk_{{chunk_id}}_output",
        sequences_dir = SEQUENCES_DIR,
        samples_to_save = OPENFOLD_CFG.get("samples_to_save", 1),
    localrule: True
    shell:
        """
        python workflow/scripts/organize_openfold_outputs.py \
            --openfold_output_dir {params.openfold_output_dir} \
            --sequences_dir {params.sequences_dir} \
            --samples_to_save {params.samples_to_save}

        rm -rf {params.openfold_output_dir}
        touch {output.organized}
        """


def aggregate_openfold_organized(wildcards):
    chunk_ids = get_openfold_chunk_ids(wildcards)
    return expand(f"{OPENFOLD_CHUNKS}/chunk_{{chunk_id}}.organized", chunk_id=chunk_ids)


rule openfold_complete:
    """Aggregate sentinel: all OpenFold chunks organized."""
    input:
        aggregate_openfold_organized,
    output:
        done = f"{OUTPUT}/.openfold_complete",
    localrule: True
    shell:
        "touch {output.done}"
