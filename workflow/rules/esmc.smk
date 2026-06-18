"""
ESM-C Stage Rules
=================
chunk_yamls_for_esmc -> run_esmc (per chunk × model size) ->
organize_esmc (per chunk × size) -> esmc_complete (per size, aggregate)

One stage per ESMC model size. `esmc.models` is any subset of {6B, 600M, 300M};
each size is driven through the same rules via the {size} wildcard, so it gets
its own jobs, per-size SLURM resources, and a `.esmc_{size}_complete` sentinel.

Encoding runs inside the ESM container via containers/run_batch_esmc.py (bound
to /opt at run time, never baked in). ESMC needs only the sequence, so it runs
INDEPENDENTLY of MSA — straight from the FASTA directory (in parallel with MSA),
the runner parsing .fasta directly. MSA-stage YAMLs are only a fallback when no
fasta_dir is set. Outputs land in sequences/{seq}/esmc/{size}/.
"""

import os as _os

ESMC_CFG    = config.get("esmc", {})
ESMC_CHUNKS = f"{OUTPUT}/esmc_chunks"
ESMC_SIZES  = ESMC_CFG.get("models", ["6B"])
# Group persistence: chunks served per GPU job (one model load per group).
ESMC_CHUNKS_PER_GROUP = max(1, int(ESMC_CFG.get("chunks_per_group", 1)))
ESMC_GROUPS = f"{ESMC_CHUNKS}/groups.tsv"

# Input source precedence: explicit yaml_dir > raw FASTA dir > MSA-stage YAMLs.
# ESMC only needs the sequence (no MSA), so FASTA wins over the MSA YAMLs even
# when MSA is enabled: ESMC then runs straight from the FASTA in parallel with
# MSA instead of waiting on .msa_complete. The MSA-YAML path is only the fallback
# when no fasta_dir is given (e.g. yaml-only reruns). The in-container runner
# parses sequences directly from .fasta (prepare_boltz_chunks.py --include_fasta).
_esmc_yaml_override = config["input"].get("yaml_dir", "")
_esmc_fasta_dir = config["input"].get("fasta_dir", "")
if _esmc_yaml_override:
    ESMC_INPUT_SOURCE, ESMC_FROM_FASTA = _esmc_yaml_override, False
elif _esmc_fasta_dir:
    ESMC_INPUT_SOURCE, ESMC_FROM_FASTA = _esmc_fasta_dir, True
elif RUN_MSA:
    ESMC_INPUT_SOURCE, ESMC_FROM_FASTA = SEQUENCES_DIR, False
else:
    ESMC_INPUT_SOURCE, ESMC_FROM_FASTA = "", True
ESMC_FASTA_FLAG = "--include_fasta" if ESMC_FROM_FASTA else ""

ESMC_PARTITION = SLURM_CFG.get("esmc", {}).get("partition", SLURM_CFG.get("partition", ""))
ESMC_ACCOUNT   = SLURM_CFG.get("account", "")

# Absolute path to the in-container batch runner (bound to /opt at run time).
ESMC_RUNNER = _os.path.abspath("containers/run_batch_esmc.py")

# Rule-local constraint (NOT global): `size` is also a wildcard in the SAE
# stage with a possibly different value set, so a global constraint here would
# clobber it. `_ESMC_SIZE_RE` is reused by every rule that carries {size}.
_ESMC_SIZE_RE = "|".join(ESMC_SIZES)


def _esmc_resource(size, key, default):
    """Per-size SLURM override (slurm.resources.esmc_<size>) → shared esmc
    defaults (slurm.resources.esmc) → `default`. ESMC-6B needs far more
    GPU/host memory than 300M, so per-size overrides matter here."""
    by_size = SLURM_CFG.get("resources", {}).get(f"esmc_{size}", {})
    if key in by_size:
        return by_size[key]
    return stage_resource("esmc", key, default)


def esmc_chunk_input(wildcards):
    """ESMC needs no MSA: source from yaml_dir or fasta_dir with no dependency so
    it runs in parallel with MSA. Only the MSA-YAML fallback (no fasta_dir) waits
    on .msa_complete."""
    inputs = {}
    if _esmc_yaml_override:
        inputs["yaml_dir"] = _esmc_yaml_override
    elif _esmc_fasta_dir:
        inputs["fasta_dir"] = _esmc_fasta_dir
    elif RUN_MSA:
        inputs["upstream_done"] = f"{OUTPUT}/.msa_complete"
    return inputs


checkpoint chunk_yamls_for_esmc:
    """Symlink the per-sequence inputs (YAML or FASTA) into chunk directories."""
    input:
        unpack(esmc_chunk_input),
    output:
        manifest = f"{ESMC_CHUNKS}/manifest.txt",
    params:
        yaml_dir = ESMC_INPUT_SOURCE,
        max_files = ESMC_CFG.get("max_files_per_job", 25),
        chunks_per_group = ESMC_CHUNKS_PER_GROUP,
        fasta_flag = ESMC_FASTA_FLAG,
    localrule: True
    shell:
        """
        python workflow/scripts/prepare_boltz_chunks.py \
            --yaml_dir {params.yaml_dir} \
            --output_dir {ESMC_CHUNKS} \
            --max_files_per_job {params.max_files} \
            --chunks_per_group {params.chunks_per_group} \
            {params.fasta_flag}
        """


def get_esmc_chunk_ids(wildcards):
    """Return chunk IDs from the ESMC manifest after the checkpoint."""
    manifest = checkpoints.chunk_yamls_for_esmc.get().output.manifest
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


def get_esmc_group_ids(wildcards):
    """Group IDs from groups.tsv after the checkpoint (one GPU job per group)."""
    checkpoints.chunk_yamls_for_esmc.get()
    return list(_read_groups(ESMC_GROUPS).keys())


def esmc_group_chunk_ids(group_id):
    """Chunk IDs belonging to one group."""
    return _read_groups(ESMC_GROUPS)[str(group_id)]


def esmc_group_chunk_dirs(wildcards):
    """Input: every chunk dir in the group (the job folds them all in one load)."""
    return [f"{ESMC_CHUNKS}/chunk_{cid}"
            for cid in esmc_group_chunk_ids(wildcards.group_id)]


rule run_esmc:
    """Encode a GROUP of chunks with one ESMC model size in a single GPU job.
    The model is loaded ONCE and serves every chunk in the group (persistence);
    the runner length-buckets all sequences across the group into padded
    micro-batches (intra-forward batching). One group sentinel drives the
    matching per-group organize_esmc."""
    wildcard_constraints:
        size = _ESMC_SIZE_RE,
        group_id = r"\d+",
    input:
        chunk_dirs = esmc_group_chunk_dirs,
    output:
        done = f"{ESMC_CHUNKS}/group_{{group_id}}_{{size}}/.done",
    benchmark:
        f"{OUTPUT}/benchmarks/esmc/encode_group{{group_id}}_{{size}}.tsv"
    log:
        f"{OUTPUT}/logs/esmc/encode_group{{group_id}}_{{size}}.log"
    params:
        chunks_root = ESMC_CHUNKS,
        chunk_ids = lambda wc: " ".join(esmc_group_chunk_ids(wc.group_id)),
        cache_dir = ESMC_CFG.get("cache_dir", ""),
        yaml_src = ESMC_INPUT_SOURCE,
        runner = ESMC_RUNNER,
        runtime = CONTAINER_RUNTIME,
        sif = container_sif("esmc"),
    resources:
        cpus_per_task   = lambda wc: _esmc_resource(wc.size, "cpus_per_task", 8),
        mem_mb          = lambda wc: _esmc_resource(wc.size, "mem_mb", 32000),
        runtime         = lambda wc: _esmc_resource(wc.size, "runtime", 60),
        esmc_jobs       = 1,
        slurm_partition = ESMC_PARTITION,
        slurm_account   = ESMC_ACCOUNT,
        slurm_extra     = slurm_extra(gpu=stage_uses_gpu("esmc", True)),
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
                 "(contains hub/models--biohub--ESMC-*)." >&2
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
            -B {params.runner}:/opt/run_batch_esmc.py:ro \
            {params.sif} \
            python /opt/run_batch_esmc.py \
                --cache /models/hf \
                --input-dirs $in_dirs \
                --output-dirs $out_dirs \
                --size {wildcards.size}

        mkdir -p "{params.chunks_root}/group_{wildcards.group_id}_{wildcards.size}"
        touch {output.done}
        """


rule organize_esmc:
    """Move a group's encoder outputs to sequences/{seq}/esmc/{size}/, then drop
    the per-chunk scratch dirs. One organize job per group mirrors the one
    run_esmc job per group."""
    wildcard_constraints:
        size = _ESMC_SIZE_RE,
        group_id = r"\d+",
    input:
        done = f"{ESMC_CHUNKS}/group_{{group_id}}_{{size}}/.done",
    output:
        organized = f"{ESMC_CHUNKS}/group_{{group_id}}_{{size}}.organized",
    params:
        chunks_root = ESMC_CHUNKS,
        chunk_ids = lambda wc: " ".join(esmc_group_chunk_ids(wc.group_id)),
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
                --stage esmc
            rm -rf "$scratch"
        done
        rm -rf "{params.chunks_root}/group_{wildcards.group_id}_{wildcards.size}"
        touch {output.organized}
        """


def aggregate_esmc_organized(wildcards):
    """Collect all .organized sentinels for one model size across groups."""
    group_ids = get_esmc_group_ids(wildcards)
    return expand(
        f"{ESMC_CHUNKS}/group_{{group_id}}_{{size}}.organized",
        group_id=group_ids, size=wildcards.size,
    )


rule esmc_complete:
    """Aggregate sentinel: all chunks encoded + organized for one model size."""
    wildcard_constraints:
        size = _ESMC_SIZE_RE,
    input:
        aggregate_esmc_organized,
    output:
        done = f"{OUTPUT}/.esmc_{{size}}_complete",
    localrule: True
    shell:
        "touch {output.done}"
