"""
ESM-C Stage Rules
=================
chunk_yamls_for_esmc -> run_esmc (per chunk × model size) ->
organize_esmc (per chunk × size) -> esmc_complete (per size, aggregate)

One stage per ESMC model size. `esmc.models` is any subset of {6B, 600M, 300M};
each size is driven through the same rules via the {size} wildcard, so it gets
its own jobs, per-size SLURM resources, and a `.esmc_{size}_complete` sentinel.

Encoding runs inside the ESM container via containers/run_batch_esmc.py (bound
to /opt at run time, never baked in). ESMC needs only the sequence, so inputs
are either the MSA-stage YAMLs (MSA on) or a raw FASTA directory (MSA off — the
runner parses .fasta directly). Outputs land in sequences/{seq}/esmc/{size}/.
"""

import os as _os

ESMC_CFG    = config.get("esmc", {})
ESMC_CHUNKS = f"{OUTPUT}/esmc_chunks"
ESMC_SIZES  = ESMC_CFG.get("models", ["6B"])

# Input source precedence: explicit yaml_dir > MSA-stage YAMLs > raw FASTA dir.
# ESMC only needs the sequence (no MSA), so the FASTA path lets it run standalone
# with MSA off — the in-container runner parses sequences straight from .fasta
# (see workflow/scripts/prepare_boltz_chunks.py --include_fasta).
_esmc_yaml_override = config["input"].get("yaml_dir", "")
_esmc_fasta_dir = config["input"].get("fasta_dir", "")
if _esmc_yaml_override:
    ESMC_INPUT_SOURCE, ESMC_FROM_FASTA = _esmc_yaml_override, False
elif RUN_MSA:
    ESMC_INPUT_SOURCE, ESMC_FROM_FASTA = SEQUENCES_DIR, False
else:
    ESMC_INPUT_SOURCE, ESMC_FROM_FASTA = _esmc_fasta_dir, True
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
    """Depend on MSA completion (so ESMC runs right after MSA, in parallel with
    Boltz). With an explicit yaml_dir or a standalone FASTA dir, no MSA dependency."""
    inputs = {}
    if _esmc_yaml_override:
        inputs["yaml_dir"] = _esmc_yaml_override
    elif RUN_MSA:
        inputs["upstream_done"] = f"{OUTPUT}/.msa_complete"
    elif _esmc_fasta_dir:
        inputs["fasta_dir"] = _esmc_fasta_dir
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
        fasta_flag = ESMC_FASTA_FLAG,
    localrule: True
    shell:
        """
        python workflow/scripts/prepare_boltz_chunks.py \
            --yaml_dir {params.yaml_dir} \
            --output_dir {ESMC_CHUNKS} \
            --max_files_per_job {params.max_files} \
            {params.fasta_flag}
        """


def get_esmc_chunk_ids(wildcards):
    """Return chunk IDs from the ESMC manifest after the checkpoint."""
    manifest = checkpoints.chunk_yamls_for_esmc.get().output.manifest
    with open(manifest) as f:
        dirs = [line.strip() for line in f if line.strip()]
    return [d.rstrip("/").split("/")[-1].replace("chunk_", "") for d in dirs]


rule run_esmc:
    """Encode a chunk of sequences with one ESMC model size, inside the ESM container."""
    wildcard_constraints:
        size = _ESMC_SIZE_RE,
        chunk_id = r"\d+",
    input:
        chunk_dir = f"{ESMC_CHUNKS}/chunk_{{chunk_id}}",
    output:
        done = f"{ESMC_CHUNKS}/chunk_{{chunk_id}}_{{size}}_output/.done",
    benchmark:
        f"{OUTPUT}/benchmarks/esmc/encode_{{chunk_id}}_{{size}}.tsv"
    log:
        f"{OUTPUT}/logs/esmc/encode_{{chunk_id}}_{{size}}.log"
    params:
        output_dir = f"{ESMC_CHUNKS}/chunk_{{chunk_id}}_{{size}}_output",
        cache_dir = ESMC_CFG.get("cache_dir", ""),
        yaml_src = ESMC_INPUT_SOURCE,
        runner = ESMC_RUNNER,
        runtime = CONTAINER_RUNTIME,
        sif = container_sif("esmc"),
    resources:
        cpus_per_task   = lambda wc: _esmc_resource(wc.size, "cpus_per_task", 8),
        mem_mb          = lambda wc: _esmc_resource(wc.size, "mem_mb", 32000),
        runtime         = lambda wc: _esmc_resource(wc.size, "runtime", 60),
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

        mkdir -p {params.output_dir}

        # Container-only: run the ESMC batch encoder inside the ESM image.
        # Mirrors containers/test/test_batch_esmc.sh — HF cache -> /models/hf
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
            -B {params.runner}:/opt/run_batch_esmc.py:ro \
            {params.sif} \
            python /opt/run_batch_esmc.py \
                --cache /models/hf \
                --input-dir {input.chunk_dir} \
                --output-dir {params.output_dir} \
                --size {wildcards.size}

        touch {output.done}
        """


rule organize_esmc:
    """Move encoder outputs to sequences/{seq}/esmc/{size}/ then drop the scratch dir."""
    wildcard_constraints:
        size = _ESMC_SIZE_RE,
        chunk_id = r"\d+",
    input:
        done = f"{ESMC_CHUNKS}/chunk_{{chunk_id}}_{{size}}_output/.done",
    output:
        organized = f"{ESMC_CHUNKS}/chunk_{{chunk_id}}_{{size}}.organized",
    params:
        scratch = f"{ESMC_CHUNKS}/chunk_{{chunk_id}}_{{size}}_output",
        sequences_dir = SEQUENCES_DIR,
    localrule: True
    shell:
        """
        python workflow/scripts/organize_encoder_outputs.py \
            --scratch_dir {params.scratch} \
            --sequences_dir {params.sequences_dir} \
            --stage esmc

        rm -rf {params.scratch}
        touch {output.organized}
        """


def aggregate_esmc_organized(wildcards):
    """Collect all .organized sentinels for one model size across chunks."""
    chunk_ids = get_esmc_chunk_ids(wildcards)
    return expand(
        f"{ESMC_CHUNKS}/chunk_{{chunk_id}}_{{size}}.organized",
        chunk_id=chunk_ids, size=wildcards.size,
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
