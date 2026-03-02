"""
ES (Evolutionary Scale / Effective Strain) Stage Rules
=======================================================
discover_es_sequences -> run_es_sequence (per sequence, SLURM) -> aggregate_es

Computes deformation metrics (effective strain, etc.) for each sequence by
averaging across multiple Boltz prediction runs using PDAnalysis's
AverageProtein. One SLURM job per sequence (no MPI needed).

Reference protein options (from config):
  es.ref_dir:  path to boltz dir with run_N/ subdirs  -> AverageProtein
  es.ref_path: path to a single CIF file              -> Protein
  es.ref_seq:  sequence name in sequences/ dir         -> resolved to ref_dir
"""

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


def resolve_ref_args():
    """Resolve reference protein config into compute_es.py arguments.

    Returns the CLI flag string for the reference protein.
    """
    ref_dir = ES_CFG.get("ref_dir", "")
    ref_path = ES_CFG.get("ref_path", "")
    ref_seq = ES_CFG.get("ref_seq", "")

    if ref_seq:
        # Use a sequence from this pipeline's output as reference
        return f"--ref_dir {SEQUENCES_DIR}/{ref_seq}/boltz"
    elif ref_dir:
        return f"--ref_dir {ref_dir}"
    elif ref_path:
        return f"--ref_path {ref_path}"
    else:
        raise ValueError(
            "ES config must specify one of: es.ref_dir, es.ref_path, or es.ref_seq"
        )


def es_input(wildcards):
    """Input for discover_es_sequences: depend on Boltz completion if enabled."""
    inputs = {}
    if RUN_BOLTZ:
        inputs["boltz_done"] = f"{OUTPUT}/.boltz_complete"
    return inputs


checkpoint discover_es_sequences:
    """Scan sequences dir and write list of sequence names to process."""
    input:
        unpack(es_input),
    output:
        seq_list = f"{ES_DIR}/sequences.txt",
    params:
        sequences_dir = SEQUENCES_DIR,
    localrule: True
    run:
        from pathlib import Path

        seq_dir = Path(params.sequences_dir)
        es_dir = Path(ES_DIR)
        es_dir.mkdir(parents=True, exist_ok=True)

        seq_names = []
        for d in sorted(seq_dir.iterdir()):
            if not d.is_dir():
                continue
            boltz_dir = d / "boltz"
            if not boltz_dir.is_dir():
                continue
            # Check for CIF files (in run_N/ subdirs or directly)
            has_cifs = (
                any(boltz_dir.glob("run_*/*.cif"))
                or any(boltz_dir.glob("*.cif"))
            )
            if has_cifs:
                seq_names.append(d.name)

        with open(output.seq_list, "w") as f:
            for name in seq_names:
                f.write(name + "\n")

        print(f"Found {len(seq_names)} sequences with CIF files for ES analysis")


def get_es_sequence_names(wildcards):
    """Return sequence names from the discover checkpoint."""
    seq_list = checkpoints.discover_es_sequences.get().output.seq_list
    with open(seq_list) as f:
        return [line.strip() for line in f if line.strip()]


rule run_es_sequence:
    """Run PDAnalysis ES analysis for one sequence (averaging across runs)."""
    input:
        seq_list = f"{ES_DIR}/sequences.txt",
    output:
        csv = f"{ES_DIR}/{{seq_name}}.csv",
    params:
        seq_boltz_dir = f"{SEQUENCES_DIR}/{{seq_name}}/boltz",
        pdanalysis_dir = PDANALYSIS_DIR,
        ref_args = resolve_ref_args(),
        method = " ".join(ES_METHOD) if isinstance(ES_METHOD, list) else ES_METHOD,
        min_plddt = ES_MIN_PLDDT,
        lddt_cutoffs = " ".join(str(c) for c in ES_LDDT_CUTOFFS) if isinstance(ES_LDDT_CUTOFFS, list) else str(ES_LDDT_CUTOFFS),
        env_path = ES_CFG.get("env_path", ""),
        container_cmd = container_cmd("pdanalysis"),
    resources:
        cpus_per_task = 4,
        mem_mb = 8000,
        runtime = 30,
        slurm_partition = ES_PARTITION,
        slurm_account = ES_ACCOUNT,
        slurm_extra = "'--gpus-per-node=1'",
    shell:
        """
        set -euo pipefail
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1

        if [ -n "{params.container_cmd}" ]; then
            {params.container_cmd} \
                --env OMP_NUM_THREADS=1 \
                --env MKL_NUM_THREADS=1 \
                --env OPENBLAS_NUM_THREADS=1 \
                python /opt/protforge/workflow/scripts/compute_es.py \
                    {params.ref_args} \
                    --seq_dir {params.seq_boltz_dir} \
                    --output {output.csv} \
                    --pdanalysis_dir /opt/pdanalysis \
                    --method {params.method} \
                    --min_plddt {params.min_plddt} \
                    --lddt_cutoffs {params.lddt_cutoffs}
        else
            module load python/3.12.8-fasrc01 gcc/14.2.0-fasrc01 || true
            set +u
            source "$(conda info --base)/etc/profile.d/conda.sh"
            if [ -n "{params.env_path}" ]; then
                conda activate {params.env_path}
            fi
            set -u
            python {workflow.basedir}/workflow/scripts/compute_es.py \
                {params.ref_args} \
                --seq_dir {params.seq_boltz_dir} \
                --output {output.csv} \
                --pdanalysis_dir {params.pdanalysis_dir} \
                --method {params.method} \
                --min_plddt {params.min_plddt} \
                --lddt_cutoffs {params.lddt_cutoffs}
        fi
        """


def aggregate_es_csvs(wildcards):
    """Collect all per-sequence CSV outputs after checkpoint expansion."""
    seq_names = get_es_sequence_names(wildcards)
    return expand(f"{ES_DIR}/{{seq_name}}.csv", seq_name=seq_names)


rule aggregate_es:
    """Combine per-sequence CSVs into combined.joblib and touch sentinel."""
    input:
        aggregate_es_csvs,
    output:
        done = f"{ES_DIR}/.done",
    params:
        es_dir = ES_DIR,
    localrule: True
    run:
        from pathlib import Path

        es_dir = Path(params.es_dir)
        csv_files = sorted(es_dir.glob("*.csv"))
        print(f"ES analysis complete: {len(csv_files)} sequence CSV files in {es_dir}")

        # Optionally combine into joblib (requires pandas + joblib)
        try:
            import joblib
            import pandas as pd
            results = {}
            for csv_path in csv_files:
                seq_name = csv_path.stem
                try:
                    results[seq_name] = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"WARNING: Failed to read {csv_path}: {e}")
            if results:
                joblib.dump(results, es_dir / "combined.joblib")
                print(f"Combined {len(results)} results into combined.joblib")
        except ImportError:
            print("pandas/joblib not available — skipping combined.joblib generation")

        Path(output.done).touch()
