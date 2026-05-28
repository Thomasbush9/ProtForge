"""
ESMFold Stage Rules
===================
checkpoint chunk_yamls_for_esmfold -> run_esmfold_chunk (per chunk) -> esmfold_complete (aggregate)

Calls slurm_scripts/run_esmfold.py to fold sequences via facebook/esmfold_v1.
Reuses workflow/scripts/chunk_yamls_for_esm.py for chunking (format-agnostic).

Smart binning: when esmfold.bin_by_length is true, the chunker partitions
sequences into a "short" pool (L < length_threshold) and a "long" pool
(L >= length_threshold), each chunked independently. Chunk wildcards then
look like `short_0`, `short_1`, ..., `long_0`, ... — the run rule's
`resources:` callable inspects the prefix and applies pool-specific mem/time.
"""

import shlex as _shlex

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

ESMFOLD_BIN_BY_LENGTH = bool(ESMFOLD_CFG.get("bin_by_length", False))
ESMFOLD_BINNING_ENABLED = bool((ESMFOLD_CFG.get("binning") or {}).get("enabled", False))
ESMFOLD_CHUNKS_TSV = f"{ESMFOLD_CHUNKS}/chunks.tsv"


# Container-mode env injection: HF_HOME override when the user pinned a host
# cache, otherwise force offline mode and use the SIF's default /models/hf
# mount point. Built once at config load; values are shell-quoted because the
# string is interpolated verbatim into the rule's bash line.
_ESMFOLD_CONTAINER_ENV = (
    f"--env HF_HOME={_shlex.quote(ESMFOLD_CFG['cache_dir'])} "
    f"--env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1"
    if ESMFOLD_CFG.get("cache_dir")
    else "--env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1"
)


def _build_esmfold_chunker_args() -> str:
    """Optional flags for chunk_yamls_for_esm.py — built once at module load."""
    parts = []
    max_seq_len = ESMFOLD_CFG.get("max_seq_len")
    if max_seq_len is not None:
        parts.append(f"--max_seq_len {int(max_seq_len)}")
    bin_args = binning_args(ESMFOLD_CFG, stage_name="esmfold")
    if bin_args:
        # New bin-aware mode supersedes the legacy 2-pool short/long path.
        parts.append(bin_args)
    elif ESMFOLD_BIN_BY_LENGTH:
        parts.append("--bin_by_length")
        parts.append(f"--length_threshold {int(ESMFOLD_CFG.get('length_threshold', 1200))}")
        default_n = int(ESMFOLD_CFG.get("num_chunks", 1))
        parts.append(f"--num_chunks_short {int(ESMFOLD_CFG.get('num_chunks_short', default_n))}")
        parts.append(f"--num_chunks_long {int(ESMFOLD_CFG.get('num_chunks_long', default_n))}")
    return " ".join(parts)


ESMFOLD_CHUNKER_EXTRA = _build_esmfold_chunker_args()


def _build_esmfold_runner_args() -> str:
    """Optional flags for run_esmfold.py."""
    parts = []
    threshold = ESMFOLD_CFG.get("chunk_size_threshold")
    if threshold is not None:
        parts.append(f"--chunk_size_threshold {int(threshold)}")
    chunk_size = ESMFOLD_CFG.get("chunk_size")
    if chunk_size is not None:
        parts.append(f"--chunk_size {int(chunk_size)}")
    return " ".join(parts)


ESMFOLD_RUNNER_EXTRA = _build_esmfold_runner_args()


def _is_long_chunk(chunk_id: str) -> bool:
    return chunk_id.startswith("long_")


def _esmfold_mem_mb(wildcards):
    # Precedence: bin-aware (chunks.tsv) > legacy short/long pools > flat.
    if ESMFOLD_BINNING_ENABLED:
        return chunk_resource(
            ESMFOLD_CHUNKS_TSV, wildcards.chunk_id, "mem_mb",
            stage_resource("esmfold", "mem_mb", 32000),
        )
    if ESMFOLD_BIN_BY_LENGTH:
        if _is_long_chunk(wildcards.chunk_id):
            return int(ESMFOLD_CFG.get("mem_long_mb", 80000))
        return int(ESMFOLD_CFG.get("mem_short_mb", 32000))
    return stage_resource("esmfold", "mem_mb", 32000)


def _esmfold_runtime(wildcards):
    if ESMFOLD_BINNING_ENABLED:
        return chunk_resource(
            ESMFOLD_CHUNKS_TSV, wildcards.chunk_id, "runtime_min",
            stage_resource("esmfold", "runtime", 120),
        )
    if ESMFOLD_BIN_BY_LENGTH:
        if _is_long_chunk(wildcards.chunk_id):
            return int(ESMFOLD_CFG.get("time_long_min", 240))
        return int(ESMFOLD_CFG.get("time_short_min", 60))
    return stage_resource("esmfold", "runtime", 120)


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
        chunker_extra = ESMFOLD_CHUNKER_EXTRA,
    localrule: True
    shell:
        """
        python workflow/scripts/chunk_yamls_for_esm.py \
            --yaml_dir {params.source_dir} \
            --output_dir {params.esmfold_chunks_dir} \
            --num_chunks {params.num_chunks} \
            --pattern '{params.glob_pattern}' \
            {params.chunker_extra}
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
    wildcard_constraints:
        chunk_id = r"(?:short_|long_)?\d+",
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
        # HF env flags injected via container_cmd's extra_env (pre-SIF
        # position required by Singularity).
        container_cmd = container_cmd("esmfold", extra_env=_ESMFOLD_CONTAINER_ENV),
        # `--cache_dir` is passed to run_esmfold.py only when explicitly set —
        # an empty string would set local_files_only=False and trigger Hub lookups.
        cache_dir_arg = (
            f"--cache_dir {_shlex.quote(ESMFOLD_CFG['cache_dir'])}"
            if ESMFOLD_CFG.get("cache_dir")
            else ""
        ),
        esmfold_chunks_dir = ESMFOLD_CHUNKS,
        runner_extra = ESMFOLD_RUNNER_EXTRA,
    resources:
        cpus_per_task = stage_resource("esmfold", "cpus_per_task", 8),
        mem_mb        = _esmfold_mem_mb,
        runtime       = _esmfold_runtime,
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
                python /opt/protforge/slurm_scripts/run_esmfold.py \
                    --fasta_list {input.fasta_list} \
                    --output_dir {params.output_dir} \
                    --processed_paths_file {params.esmfold_chunks_dir}/processed_paths_{wildcards.chunk_id}.txt \
                    {params.cache_dir_arg} \
                    {params.runner_extra}
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
                --cache_dir {params.cache_dir} \
                {params.runner_extra}
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
