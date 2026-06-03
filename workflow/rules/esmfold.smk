"""
ESMFold2 Stage Rules
====================
chunk_yamls_for_esmfold -> run_esmfold (per chunk) ->
organize_esmfold (per chunk) -> esmfold_complete (aggregate)

Folds the per-sequence YAMLs from the MSA stage with ESMFold2 (biohub
"fast" variant) inside the ESM container via containers/run_batch_esmfold.py
(bound to /opt at run time, never baked in). Outputs (structure.cif, plddt.npy,
metrics.pt) land in sequences/{seq}/esmfold/fast/.
"""

import os as _os

ESMFOLD_CFG    = config.get("esmfold", {})
ESMFOLD_CHUNKS = f"{OUTPUT}/esmfold_chunks"

# YAML source: user-provided yaml_dir, or the per-sequence YAMLs from MSA.
_esmfold_yaml_override = config["input"].get("yaml_dir", "")
ESMFOLD_YAML_SOURCE = _esmfold_yaml_override if _esmfold_yaml_override else SEQUENCES_DIR

ESMFOLD_PARTITION = SLURM_CFG.get("esmfold", {}).get("partition", SLURM_CFG.get("partition", ""))
ESMFOLD_ACCOUNT   = SLURM_CFG.get("account", "")

# Absolute path to the in-container batch runner (bound to /opt at run time).
ESMFOLD_RUNNER = _os.path.abspath("containers/run_batch_esmfold.py")


wildcard_constraints:
    chunk_id = r"\d+",


def esmfold_chunk_input(wildcards):
    """Depend on MSA completion (so ESMFold2 runs right after MSA, in parallel
    with Boltz). When the user supplies their own yaml_dir, no dependency."""
    inputs = {}
    if _esmfold_yaml_override:
        inputs["yaml_dir"] = _esmfold_yaml_override
    elif RUN_MSA:
        inputs["upstream_done"] = f"{OUTPUT}/.msa_complete"
    return inputs


checkpoint chunk_yamls_for_esmfold:
    """Symlink the per-sequence YAMLs into chunk directories for parallel folding."""
    input:
        unpack(esmfold_chunk_input),
    output:
        manifest = f"{ESMFOLD_CHUNKS}/manifest.txt",
    params:
        yaml_dir = ESMFOLD_YAML_SOURCE,
        max_files = ESMFOLD_CFG.get("max_files_per_job", 25),
    localrule: True
    shell:
        """
        python workflow/scripts/prepare_boltz_chunks.py \
            --yaml_dir {params.yaml_dir} \
            --output_dir {ESMFOLD_CHUNKS} \
            --max_files_per_job {params.max_files}
        """


def get_esmfold_chunk_ids(wildcards):
    """Return chunk IDs from the ESMFold2 manifest after the checkpoint."""
    manifest = checkpoints.chunk_yamls_for_esmfold.get().output.manifest
    with open(manifest) as f:
        dirs = [line.strip() for line in f if line.strip()]
    return [d.rstrip("/").split("/")[-1].replace("chunk_", "") for d in dirs]


rule run_esmfold:
    """Fold a chunk of sequences with ESMFold2, inside the ESM container."""
    input:
        chunk_dir = f"{ESMFOLD_CHUNKS}/chunk_{{chunk_id}}",
    output:
        done = f"{ESMFOLD_CHUNKS}/chunk_{{chunk_id}}_output/.done",
    benchmark:
        f"{OUTPUT}/benchmarks/esmfold/fold_{{chunk_id}}.tsv"
    log:
        f"{OUTPUT}/logs/esmfold/fold_{{chunk_id}}.log"
    params:
        output_dir = f"{ESMFOLD_CHUNKS}/chunk_{{chunk_id}}_output",
        cache_dir = ESMFOLD_CFG.get("cache_dir", ""),
        yaml_src = ESMFOLD_YAML_SOURCE,
        runner = ESMFOLD_RUNNER,
        num_loops = ESMFOLD_CFG.get("num_loops", 3),
        num_sampling_steps = ESMFOLD_CFG.get("num_sampling_steps", 50),
        seed = ESMFOLD_CFG.get("seed", 0),
        runtime = CONTAINER_RUNTIME,
        sif = container_sif("esmfold"),
    resources:
        cpus_per_task   = stage_resource("esmfold", "cpus_per_task", 8),
        mem_mb          = stage_resource("esmfold", "mem_mb", 32000),
        runtime         = stage_resource("esmfold", "runtime", 120),
        slurm_partition = ESMFOLD_PARTITION,
        slurm_account   = ESMFOLD_ACCOUNT,
        slurm_extra     = slurm_extra(gpu=stage_uses_gpu("esmfold", True)),
    shell:
        """
        set -euo pipefail
        exec > {log} 2>&1
        set -x

        export CUDA_VISIBLE_DEVICES=0

        if [ -z "{params.sif}" ]; then
            echo "ERROR: no ESM container configured. Set containers.esmfold" \
                 "(or containers.gpu) to the ESM .sif path." >&2
            exit 1
        fi
        if [ -z "{params.cache_dir}" ]; then
            echo "ERROR: esmfold.cache_dir must point to the host HF cache" \
                 "(contains hub/models--biohub--ESMFold2-Fast)." >&2
            exit 1
        fi

        mkdir -p {params.output_dir}

        # Container-only: run the ESMFold2 batch runner inside the ESM image.
        # Mirrors containers/test/esmfold2_test_image.sh — HF cache -> /models/hf
        # (ro); chunk dir + sequences/ bound same-path so the symlinked YAMLs
        # resolve; runner bound to /opt; --cleanenv + offline HF flags.
        {params.runtime} exec --nv --cleanenv \
            --env HF_HOME=/models/hf \
            --env HF_HUB_OFFLINE=1 \
            --env TRANSFORMERS_OFFLINE=1 \
            --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
            -B {params.cache_dir}:/models/hf:ro \
            -B {input.chunk_dir}:{input.chunk_dir}:ro \
            -B {params.yaml_src}:{params.yaml_src}:ro \
            -B {params.output_dir}:{params.output_dir} \
            -B {params.runner}:/opt/run_batch_esmfold.py:ro \
            {params.sif} \
            python /opt/run_batch_esmfold.py \
                --cache /models/hf \
                --input-dir {input.chunk_dir} \
                --output-dir {params.output_dir} \
                --num-loops {params.num_loops} \
                --num-sampling-steps {params.num_sampling_steps} \
                --seed {params.seed}

        touch {output.done}
        """


rule organize_esmfold:
    """Move fold outputs to sequences/{seq}/esmfold/fast/ then drop the scratch dir."""
    input:
        done = f"{ESMFOLD_CHUNKS}/chunk_{{chunk_id}}_output/.done",
    output:
        organized = f"{ESMFOLD_CHUNKS}/chunk_{{chunk_id}}.organized",
    params:
        scratch = f"{ESMFOLD_CHUNKS}/chunk_{{chunk_id}}_output",
        sequences_dir = SEQUENCES_DIR,
    localrule: True
    shell:
        """
        python workflow/scripts/organize_encoder_outputs.py \
            --scratch_dir {params.scratch} \
            --sequences_dir {params.sequences_dir} \
            --stage esmfold

        rm -rf {params.scratch}
        touch {output.organized}
        """


def aggregate_esmfold_organized(wildcards):
    """Collect all .organized sentinels across chunks."""
    chunk_ids = get_esmfold_chunk_ids(wildcards)
    return expand(f"{ESMFOLD_CHUNKS}/chunk_{{chunk_id}}.organized", chunk_id=chunk_ids)


rule esmfold_complete:
    """Aggregate sentinel: all ESMFold2 chunks folded + organized."""
    input:
        aggregate_esmfold_organized,
    output:
        done = f"{OUTPUT}/.esmfold_complete",
    localrule: True
    shell:
        "touch {output.done}"
