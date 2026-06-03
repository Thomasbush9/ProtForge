"""
ESM-C SAE Stage Rules
=====================
chunk_yamls_for_sae -> run_esmc_sae (per chunk × model size) ->
organize_esmc_sae (per chunk × size) -> esmc_sae_complete (per size, aggregate)

Extracts sparse-autoencoder activations from ESMC. SAEs are model-size specific
(biohub/ESMC-<size>-sae-...), so this mirrors the ESMC embedding stage's {size}
wildcard — one job per size, each extracting all requested layers in a single
forward pass (`esmc.sae.layers`, "all" = every trained layer).

SAE recomputes from the sequence (loads the ESMC model + SAE, forward pass), so
it depends on the MSA-stage YAMLs, NOT on the embedding outputs — it runs
standalone even when embeddings were produced in an earlier run. Gated by
`esmc.sae.enabled`, independent of `pipeline.esmc`. Encoding runs inside the ESM
container via containers/esmc_extract_sae.py (bound to /opt at run time).
Outputs land in sequences/{seq}/sae/{size}/{sae_type}/.
"""

import os as _os

ESMC_SAE_CFG    = config.get("esmc", {}).get("sae", {})
ESMC_SAE_CHUNKS = f"{OUTPUT}/esmc_sae_chunks"
# Sizes to extract SAE for: explicit list, else fall back to esmc.models.
ESMC_SAE_SIZES  = ESMC_SAE_CFG.get("sizes") or config.get("esmc", {}).get("models", ["6B"])
ESMC_SAE_TYPE   = ESMC_SAE_CFG.get("sae_type", "all-layers")
ESMC_SAE_LAYERS = str(ESMC_SAE_CFG.get("layers", "all"))

# YAML source: user-provided yaml_dir, or the per-sequence YAMLs from MSA.
_sae_yaml_override = config["input"].get("yaml_dir", "")
ESMC_SAE_YAML_SOURCE = _sae_yaml_override if _sae_yaml_override else SEQUENCES_DIR

ESMC_SAE_PARTITION = SLURM_CFG.get("esmc_sae", {}).get("partition", SLURM_CFG.get("partition", ""))
ESMC_SAE_ACCOUNT   = SLURM_CFG.get("account", "")

# Absolute path to the in-container extractor (bound to /opt at run time).
ESMC_SAE_RUNNER = _os.path.abspath("containers/esmc_extract_sae.py")

# Rule-local constraint (NOT global) to avoid clobbering the embedding stage's
# `size` constraint when both are included with different size sets.
_ESMC_SAE_SIZE_RE = "|".join(ESMC_SAE_SIZES)


def _esmc_sae_resource(size, key, default):
    """Per-size SLURM override (slurm.resources.esmc_sae_<size>) → shared
    esmc_sae defaults (slurm.resources.esmc_sae) → `default`."""
    by_size = SLURM_CFG.get("resources", {}).get(f"esmc_sae_{size}", {})
    if key in by_size:
        return by_size[key]
    return stage_resource("esmc_sae", key, default)


def esmc_sae_chunk_input(wildcards):
    """Depend on MSA completion (runs right after MSA, in parallel with Boltz /
    ESMC embeddings). With a user-provided yaml_dir there's no dependency."""
    inputs = {}
    if _sae_yaml_override:
        inputs["yaml_dir"] = _sae_yaml_override
    elif RUN_MSA:
        inputs["upstream_done"] = f"{OUTPUT}/.msa_complete"
    return inputs


checkpoint chunk_yamls_for_sae:
    """Symlink the per-sequence YAMLs into chunk directories for parallel extract."""
    input:
        unpack(esmc_sae_chunk_input),
    output:
        manifest = f"{ESMC_SAE_CHUNKS}/manifest.txt",
    params:
        yaml_dir = ESMC_SAE_YAML_SOURCE,
        max_files = ESMC_SAE_CFG.get("max_files_per_job", 25),
    localrule: True
    shell:
        """
        python workflow/scripts/prepare_boltz_chunks.py \
            --yaml_dir {params.yaml_dir} \
            --output_dir {ESMC_SAE_CHUNKS} \
            --max_files_per_job {params.max_files}
        """


def get_esmc_sae_chunk_ids(wildcards):
    """Return chunk IDs from the SAE manifest after the checkpoint."""
    manifest = checkpoints.chunk_yamls_for_sae.get().output.manifest
    with open(manifest) as f:
        dirs = [line.strip() for line in f if line.strip()]
    return [d.rstrip("/").split("/")[-1].replace("chunk_", "") for d in dirs]


rule run_esmc_sae:
    """Extract SAE activations for a chunk with one ESMC model size, in the ESM container."""
    wildcard_constraints:
        size = _ESMC_SAE_SIZE_RE,
        chunk_id = r"\d+",
    input:
        chunk_dir = f"{ESMC_SAE_CHUNKS}/chunk_{{chunk_id}}",
    output:
        done = f"{ESMC_SAE_CHUNKS}/chunk_{{chunk_id}}_{{size}}_output/.done",
    benchmark:
        f"{OUTPUT}/benchmarks/esmc_sae/extract_{{chunk_id}}_{{size}}.tsv"
    log:
        f"{OUTPUT}/logs/esmc_sae/extract_{{chunk_id}}_{{size}}.log"
    params:
        output_dir = f"{ESMC_SAE_CHUNKS}/chunk_{{chunk_id}}_{{size}}_output",
        cache_dir = config.get("esmc", {}).get("cache_dir", ""),
        yaml_src = ESMC_SAE_YAML_SOURCE,
        runner = ESMC_SAE_RUNNER,
        sae_type = ESMC_SAE_TYPE,
        layers = ESMC_SAE_LAYERS,
        runtime = CONTAINER_RUNTIME,
        sif = container_sif("esmc"),
    resources:
        cpus_per_task   = lambda wc: _esmc_sae_resource(wc.size, "cpus_per_task", 8),
        mem_mb          = lambda wc: _esmc_sae_resource(wc.size, "mem_mb", 32000),
        runtime         = lambda wc: _esmc_sae_resource(wc.size, "runtime", 60),
        slurm_partition = ESMC_SAE_PARTITION,
        slurm_account   = ESMC_SAE_ACCOUNT,
        slurm_extra     = slurm_extra(gpu=stage_uses_gpu("esmc_sae", True)),
    shell:
        """
        set -euo pipefail
        exec > {log} 2>&1
        set -x

        export CUDA_VISIBLE_DEVICES=0

        if [ -z "{params.sif}" ]; then
            echo "ERROR: no ESM container configured. Set containers.esmc" \
                 "(or containers.gpu) to the ESM .sif path." >&2
            exit 1
        fi
        if [ -z "{params.cache_dir}" ]; then
            echo "ERROR: esmc.cache_dir must point to the host HF cache" \
                 "(contains the ESMC model + SAE repos)." >&2
            exit 1
        fi

        mkdir -p {params.output_dir}

        # Container-only: run the SAE extractor inside the ESM image.
        # Mirrors containers/test/test_esmc_extract_sae.sh — HF cache -> /models/hf
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
            -B {params.runner}:/opt/esmc_extract_sae.py:ro \
            {params.sif} \
            python /opt/esmc_extract_sae.py \
                --cache /models/hf \
                --input-dir {input.chunk_dir} \
                --output-dir {params.output_dir} \
                --size {wildcards.size} \
                --sae {params.sae_type} \
                --layers {params.layers}

        touch {output.done}
        """


rule organize_esmc_sae:
    """Move SAE outputs to sequences/{seq}/sae/{size}/ then drop the scratch dir."""
    wildcard_constraints:
        size = _ESMC_SAE_SIZE_RE,
        chunk_id = r"\d+",
    input:
        done = f"{ESMC_SAE_CHUNKS}/chunk_{{chunk_id}}_{{size}}_output/.done",
    output:
        organized = f"{ESMC_SAE_CHUNKS}/chunk_{{chunk_id}}_{{size}}.organized",
    params:
        scratch = f"{ESMC_SAE_CHUNKS}/chunk_{{chunk_id}}_{{size}}_output",
        sequences_dir = SEQUENCES_DIR,
    localrule: True
    shell:
        """
        python workflow/scripts/organize_encoder_outputs.py \
            --scratch_dir {params.scratch} \
            --sequences_dir {params.sequences_dir} \
            --stage sae

        rm -rf {params.scratch}
        touch {output.organized}
        """


def aggregate_esmc_sae_organized(wildcards):
    """Collect all .organized sentinels for one model size across chunks."""
    chunk_ids = get_esmc_sae_chunk_ids(wildcards)
    return expand(
        f"{ESMC_SAE_CHUNKS}/chunk_{{chunk_id}}_{{size}}.organized",
        chunk_id=chunk_ids, size=wildcards.size,
    )


rule esmc_sae_complete:
    """Aggregate sentinel: all chunks extracted + organized for one model size."""
    wildcard_constraints:
        size = _ESMC_SAE_SIZE_RE,
    input:
        aggregate_esmc_sae_organized,
    output:
        done = f"{OUTPUT}/.esmc_sae_{{size}}_complete",
    localrule: True
    shell:
        "touch {output.done}"
