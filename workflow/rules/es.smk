"""
ES (Evolutionary Scale / Effective Strain) Stage Rules
=======================================================
Two rules:
  1. collect_es_paths (localrule) — scans boltz dirs, writes paths.txt files
  2. run_es_all (SLURM job)       — runs PDAnalysis for each sequence

Both call standalone Python scripts with explicit conda env python:
  {env_path}/bin/python script.py ...
This avoids all python interpreter issues — no module load, no conda activate,
no PATH ambiguity.

Reference protein options (from config):
  es.ref_dir:  path to boltz dir with run_N/ subdirs  -> AverageProtein
  es.ref_path: path to a single CIF file              -> Protein
  es.ref_seq:  sequence name in sequences/ dir         -> resolved to ref_dir
"""

import sys as _sys

ES_CFG = config.get("es", {})
ES_DIR = f"{OUTPUT}/es"

ES_PARTITION = SLURM_CFG.get("es", {}).get("partition", SLURM_CFG.get("partition", ""))
ES_ACCOUNT = SLURM_CFG.get("account", "")

# PDAnalysis directory (support both pdanalysis_dir and legacy script_dir)
PDANALYSIS_DIR = ES_CFG.get("pdanalysis_dir", ES_CFG.get("script_dir", ""))

# ES deformation parameters
ES_METHOD = ES_CFG.get("method", ["strain"])
ES_MIN_PLDDT = ES_CFG.get("min_plddt", 70)
ES_LDDT_CUTOFFS = ES_CFG.get("lddt_cutoffs", [0.125, 0.25, 0.5, 1])

# Conda env python — used directly to avoid interpreter mismatches
ES_ENV_PATH = ES_CFG.get("env_path", "")
ES_PYTHON = f"{ES_ENV_PATH}/bin/python" if ES_ENV_PATH else "python"


def _resolve_ref_source():
    """Resolve reference config to a path (directory or file)."""
    ref_seq = ES_CFG.get("ref_seq", "")
    ref_dir = ES_CFG.get("ref_dir", "")
    ref_path = ES_CFG.get("ref_path", "")

    if ref_seq:
        return f"{SEQUENCES_DIR}/{ref_seq}/boltz"
    elif ref_dir:
        return ref_dir
    elif ref_path:
        return ref_path
    else:
        raise ValueError(
            "ES is enabled (pipeline.es: true) but no reference structure is configured.\n"
            "Set one of these in config.yaml:\n"
            "  es.ref_path: /path/to/wildtype.cif        (single CIF file)\n"
            "  es.ref_dir:  /path/to/boltz/dir           (directory with run_N/ subdirs)\n"
            "  es.ref_seq:  sequence_name                 (name from this pipeline's output)\n"
            "Or disable ES with: pipeline.es: false"
        )


def _ref_is_single_file():
    """Check if reference is a single file (ref_path) vs directory."""
    ref_path = ES_CFG.get("ref_path", "")
    ref_dir = ES_CFG.get("ref_dir", "")
    ref_seq = ES_CFG.get("ref_seq", "")
    return bool(ref_path) and not bool(ref_seq) and not bool(ref_dir)


def _ref_cli_flag():
    """Return the CLI flag for run_es.py: --ref_path or --ref_dir."""
    return "--ref_path" if _ref_is_single_file() else "--ref_dir"


def _collect_input(wildcards):
    """Input for collect_es_paths: depend on Boltz completion if enabled."""
    inputs = {}
    if RUN_BOLTZ:
        inputs["boltz_done"] = f"{OUTPUT}/.boltz_complete"
    return inputs


def _es_input(wildcards):
    """Input for run_es_all: depend on path collection."""
    return {"paths_done": f"{ES_DIR}/.paths_collected"}


# ---------------------------------------------------------------------------
# Rule 1: Collect CIF paths (runs on login node)
# ---------------------------------------------------------------------------

rule collect_es_paths:
    """Scan boltz output dirs and write paths.txt for each sequence."""
    input:
        unpack(_collect_input),
    output:
        done = f"{ES_DIR}/.paths_collected",
    params:
        python = _sys.executable,
        sequences_dir = SEQUENCES_DIR,
        ref_source = _resolve_ref_source(),
        ref_is_file = _ref_is_single_file(),
    localrule: True
    shell:
        """
        set -euo pipefail

        # Collect paths for all sequences
        {params.python} workflow/scripts/collect_cif_paths.py \
            --sequences_dir {params.sequences_dir}

        # If reference is a directory, collect its paths too
        if [ "{params.ref_is_file}" = "False" ]; then
            {params.python} workflow/scripts/collect_cif_paths.py \
                --boltz_dir {params.ref_source} \
                --output {params.ref_source}/paths.txt
        fi

        touch {output.done}
        """


# ---------------------------------------------------------------------------
# Rule 2: Run ES analysis (SLURM job)
# ---------------------------------------------------------------------------

rule run_es_all:
    """Run PDAnalysis ES for all sequences in a single SLURM job."""
    input:
        unpack(_es_input),
    output:
        done = f"{ES_DIR}/.done",
    benchmark:
        f"{OUTPUT}/benchmarks/es/es_all.tsv"
    log:
        f"{OUTPUT}/logs/es/es_all.log",
    params:
        es_python = ES_PYTHON,
        sequences_dir = SEQUENCES_DIR,
        es_dir = ES_DIR,
        pdanalysis_dir = PDANALYSIS_DIR,
        ref_flag = _ref_cli_flag(),
        ref_source = _resolve_ref_source(),
        method = " ".join(ES_METHOD) if isinstance(ES_METHOD, list) else ES_METHOD,
        min_plddt = ES_MIN_PLDDT,
        lddt_cutoffs = " ".join(str(c) for c in ES_LDDT_CUTOFFS) if isinstance(ES_LDDT_CUTOFFS, list) else str(ES_LDDT_CUTOFFS),
    resources:
        cpus_per_task = 4,
        mem_mb = 16000,
        runtime = 120,
        slurm_partition = ES_PARTITION,
        slurm_account = ES_ACCOUNT,
        slurm_extra = slurm_extra(gpu=True),
    shell:
        """
        set -euo pipefail
        exec > {log} 2>&1
        set -x

        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1

        {params.es_python} workflow/scripts/run_es.py \
            {params.ref_flag} {params.ref_source} \
            --sequences_dir {params.sequences_dir} \
            --output_dir {params.es_dir} \
            --pdanalysis_dir {params.pdanalysis_dir} \
            --method {params.method} \
            --min_plddt {params.min_plddt} \
            --lddt_cutoffs {params.lddt_cutoffs}

        touch {output.done}
        """
