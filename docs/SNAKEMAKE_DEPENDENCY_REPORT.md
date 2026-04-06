# Snakemake Repo Cleanup Report

Scope: repo overview for the Slurm/Snakemake path:

```bash
snakemake --profile profiles/slurm/
```

Assumption: on the cluster you may enable all core stages:

```yaml
pipeline:
  msa: true
  boltz: true
  esm: true
  es: true
```

Local verification note: `snakemake` is not on this workstation's `PATH`, and the `/n/...` cluster paths are not mounted here. This report is a static trace from `Snakefile`, rule files, and script references.

## Keep For Full Snakemake Runtime

These are needed by the full `snakemake --profile profiles/slurm/` workflow.

Core entry/config:

- `Snakefile`
- `config.yaml`
- `profiles/slurm/config.yaml`

Snakemake rule files:

- `workflow/rules/msa.smk`
- `workflow/rules/boltz.smk`
- `workflow/rules/esm.smk`
- `workflow/rules/es.smk`

Workflow helper scripts called by those rules:

- `workflow/scripts/chunk_fastas.py`
- `workflow/scripts/organize_msa_outputs.py`
- `workflow/scripts/prepare_boltz_chunks.py`
- `workflow/scripts/organize_boltz_outputs.py`
- `workflow/scripts/chunk_yamls_for_esm.py`
- `workflow/scripts/collect_cif_paths.py`
- `workflow/scripts/run_es.py`

Slurm script still called by the Snakemake workflow:

- `slurm_scripts/run_esm.py`

Python package code needed by `slurm_scripts/run_esm.py`:

- `utils/utils.py`
- `utils/__init__.py`

Optional runtime support:

- `config.template.yaml`, if you want a maintained config reference.
- Container documentation/config paths if you plan to use `config.containers.*`; the current Snakemake code supports containers through `container_cmd()`.

## Keep, But Not Runtime-Critical

These are not needed by Snakemake execution itself, but they serve useful repo roles.

Setup and docs:

- `setup.sh`
- `README.md`
- `docs/CLUSTER_SETUP.md`
- `docs/SNAKEMAKE_GUIDE.md`
- `docs/CONTAINERS.md`
- `docs/SNAKEMAKE_DEPENDENCY_REPORT.md`

Data preparation:

- `bash_scripts/generate_data.sh`
- `utils/generate_data.py`
- `utils/generate_subsamples.py`
- `requirements-data.txt`

Tests and package metadata:

- `tests/`
- `pyproject.toml`

Visualization:

- `viz/`

Developer notes:

- `TODO.md`, if still useful.
- `CLAUDE.md`, if still used by your tooling.

## Strong Cleanup Candidates

These are not called by `Snakefile` or `workflow/rules/*.smk` in the full Snakemake workflow. They look like older manual Slurm orchestration paths that have been replaced by the Snakemake rules.

Root-level manual launchers:

- `run.sh`
- `run_msa.sh`
- `run_boltz.sh`
- `run_boltz_predictions.sh`
- `run_esm_standalone.sh`
- `run_es_standalone.sh`

Manual/download helper:

- `download_tools.sh`

Legacy Slurm wrappers and arrays:

- `slurm_scripts/run_msa_array.slrm`
- `slurm_scripts/run_boltz_array.slrm`
- `slurm_scripts/run_boltz_organize.slrm`
- `slurm_scripts/run_boltz_wrapper.slrm`
- `slurm_scripts/run_esm_array.slrm`
- `slurm_scripts/run_esm_wrapper.slrm`
- `slurm_scripts/run_es_array.slrm`

Legacy stage splitters/runners:

- `slurm_scripts/split_and_run_msa.sh`
- `slurm_scripts/split_and_run_boltz.sh`
- `slurm_scripts/run_esm.sh`
- `slurm_scripts/run_es.sh`
- `slurm_scripts/process_msa_fasta.sh`
- `slurm_scripts/organize_boltz_outputs.sh`

Legacy checker/retry path:

- `slurm_scripts/checker.sh`
- `slurm_scripts/checker_msa.sh`
- `slurm_scripts/checker_boltz.sh`
- `slurm_scripts/checker_esm.sh`
- `slurm_scripts/run_checker_msa.slrm`
- `slurm_scripts/run_checker_boltz.slrm`
- `slurm_scripts/run_checker_esm.slrm`

Other legacy/manual helper:

- `slurm_scripts/example.sh`
- `slurm_scripts/parse_config.py`

Important caveat: keep `slurm_scripts/run_esm.py`. It is still used by `workflow/rules/esm.smk`.

## Generated Files To Remove

These should not be versioned or kept around as repo source:

- `.pytest_cache/`
- `__pycache__/` directories
- `*.pyc`

The current `.gitignore` already ignores `*__pycache__`, `__pycache__/*`, and `*pyc`. It also ignores `.pytest_cache/.gitignore` only because of the nested `.gitignore`, not the whole cache directory. Consider adding:

```gitignore
.pytest_cache/
```

## Directory-Level Summary

Needed by full Snakemake runtime:

- `profiles/slurm/`
- `workflow/rules/`
- `workflow/scripts/`
- `slurm_scripts/` only for `run_esm.py`
- `utils/` only for `utils.py` and `__init__.py`

Not needed by full Snakemake runtime, but useful for non-runtime repo roles:

- `bash_scripts/` for data prep
- `tests/` for validation
- `docs/` for documentation
- `viz/` for visualization

Likely simplification target:

- Most of `slurm_scripts/` except `run_esm.py`
- Most root-level `run*.sh` launchers if Snakemake is now the supported execution path

## Suggested Cleanup Plan

1. Keep the Snakemake runtime files listed above.
2. Move the strong cleanup candidates to an archive branch or delete them in one PR after confirming nobody still uses standalone/manual launchers.
3. Update `README.md` and `docs/SNAKEMAKE_GUIDE.md` to point users only to `snakemake --profile profiles/slurm/` if you remove the legacy launchers.
4. Add `.pytest_cache/` to `.gitignore`.
5. Keep tests before deleting old scripts; they cover the active `workflow/scripts/*.py` helpers and are useful for preventing regressions.
