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
# Group persistence: chunks served per GPU job (one model load per group).
ESMC_SAE_CHUNKS_PER_GROUP = max(1, int(ESMC_SAE_CFG.get("chunks_per_group", 1)))
ESMC_SAE_GROUPS = f"{ESMC_SAE_CHUNKS}/groups.tsv"

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
        chunks_per_group = ESMC_SAE_CHUNKS_PER_GROUP,
    localrule: True
    shell:
        """
        python workflow/scripts/prepare_boltz_chunks.py \
            --yaml_dir {params.yaml_dir} \
            --output_dir {ESMC_SAE_CHUNKS} \
            --max_files_per_job {params.max_files} \
            --chunks_per_group {params.chunks_per_group}
        """


def get_esmc_sae_chunk_ids(wildcards):
    """Return chunk IDs from the SAE manifest after the checkpoint."""
    manifest = checkpoints.chunk_yamls_for_sae.get().output.manifest
    with open(manifest) as f:
        dirs = [line.strip() for line in f if line.strip()]
    return [d.rstrip("/").split("/")[-1].replace("chunk_", "") for d in dirs]


def _read_groups(groups_tsv):
    """Parse groups.tsv -> {group_id: [chunk_id, ...]}."""
    out = {}
    with open(groups_tsv) as f:
        next(f, None)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            gid, cids = line.split("\t")
            out[gid] = cids.split(",")
    return out


def get_esmc_sae_group_ids(wildcards):
    """Group IDs from groups.tsv after the checkpoint (one GPU job per group)."""
    checkpoints.chunk_yamls_for_sae.get()
    return list(_read_groups(ESMC_SAE_GROUPS).keys())


def esmc_sae_group_chunk_ids(group_id):
    """Chunk IDs belonging to one group."""
    return _read_groups(ESMC_SAE_GROUPS)[str(group_id)]


def esmc_sae_group_chunk_dirs(wildcards):
    """Input: every chunk dir in the group (the job folds them all in one load)."""
    return [f"{ESMC_SAE_CHUNKS}/chunk_{cid}"
            for cid in esmc_sae_group_chunk_ids(wildcards.group_id)]


rule run_esmc_sae:
    """Extract SAE activations for a GROUP of chunks with one ESMC model size in
    a single GPU job. The model + SAE are loaded ONCE and serve every chunk in
    the group (persistence); the extractor length-buckets all sequences across
    the group into padded micro-batches. One group sentinel drives the matching
    per-group organize_esmc_sae."""
    wildcard_constraints:
        size = _ESMC_SAE_SIZE_RE,
        group_id = r"\d+",
    input:
        chunk_dirs = esmc_sae_group_chunk_dirs,
    output:
        done = f"{ESMC_SAE_CHUNKS}/group_{{group_id}}_{{size}}/.done",
    benchmark:
        f"{OUTPUT}/benchmarks/esmc_sae/extract_group{{group_id}}_{{size}}.tsv"
    log:
        f"{OUTPUT}/logs/esmc_sae/extract_group{{group_id}}_{{size}}.log"
    params:
        chunks_root = ESMC_SAE_CHUNKS,
        chunk_ids = lambda wc: " ".join(esmc_sae_group_chunk_ids(wc.group_id)),
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

        # Build the parallel --input-dirs / --output-dirs lists for this group.
        in_dirs=""; out_dirs=""
        for cid in {params.chunk_ids}; do
            in_dirs="$in_dirs {params.chunks_root}/chunk_${{cid}}"
            od="{params.chunks_root}/chunk_${{cid}}_{wildcards.size}_output"
            out_dirs="$out_dirs $od"
            mkdir -p "$od"
        done

        # Bind the whole chunks_root once (covers every chunk dir + output dir);
        # sequences/ bound same-path so symlinked YAMLs resolve; runner -> /opt.
        {params.runtime} exec --nv --cleanenv \
            --env HF_HOME=/models/hf \
            --env HF_HUB_OFFLINE=1 \
            --env TRANSFORMERS_OFFLINE=1 \
            --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
            -B {params.cache_dir}:/models/hf:ro \
            -B {params.chunks_root}:{params.chunks_root} \
            -B {params.yaml_src}:{params.yaml_src}:ro \
            -B {params.runner}:/opt/esmc_extract_sae.py:ro \
            {params.sif} \
            python /opt/esmc_extract_sae.py \
                --cache /models/hf \
                --input-dirs $in_dirs \
                --output-dirs $out_dirs \
                --size {wildcards.size} \
                --sae {params.sae_type} \
                --layers {params.layers}

        mkdir -p "{params.chunks_root}/group_{wildcards.group_id}_{wildcards.size}"
        touch {output.done}
        """


rule organize_esmc_sae:
    """Move a group's SAE outputs to sequences/{seq}/sae/{size}/, then drop the
    per-chunk scratch dirs. One organize job per group mirrors the one
    run_esmc_sae job per group."""
    wildcard_constraints:
        size = _ESMC_SAE_SIZE_RE,
        group_id = r"\d+",
    input:
        done = f"{ESMC_SAE_CHUNKS}/group_{{group_id}}_{{size}}/.done",
    output:
        organized = f"{ESMC_SAE_CHUNKS}/group_{{group_id}}_{{size}}.organized",
    params:
        chunks_root = ESMC_SAE_CHUNKS,
        chunk_ids = lambda wc: " ".join(esmc_sae_group_chunk_ids(wc.group_id)),
        sequences_dir = SEQUENCES_DIR,
    localrule: True
    shell:
        """
        set -euo pipefail
        for cid in {params.chunk_ids}; do
            scratch="{params.chunks_root}/chunk_${{cid}}_{wildcards.size}_output"
            python workflow/scripts/organize_encoder_outputs.py \
                --scratch_dir "$scratch" \
                --sequences_dir {params.sequences_dir} \
                --stage sae
            rm -rf "$scratch"
        done
        rm -rf "{params.chunks_root}/group_{wildcards.group_id}_{wildcards.size}"
        touch {output.organized}
        """


def aggregate_esmc_sae_organized(wildcards):
    """Collect all .organized sentinels for one model size across groups."""
    group_ids = get_esmc_sae_group_ids(wildcards)
    return expand(
        f"{ESMC_SAE_CHUNKS}/group_{{group_id}}_{{size}}.organized",
        group_id=group_ids, size=wildcards.size,
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
