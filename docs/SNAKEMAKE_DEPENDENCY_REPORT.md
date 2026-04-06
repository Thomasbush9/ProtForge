# Snakemake Dependency Report

Scope: `snakemake --profile profiles/slurm/` from the repo root, using the current `config.yaml`.

Local verification note: `snakemake` is not on this workstation's `PATH`, and the `/n/...` cluster paths in `config.yaml` are not mounted here. This report is therefore a static dependency trace from `Snakefile`, the active config flags, and the rule files.

## Active config path

Current `config.yaml` sets:

```yaml
pipeline:
  msa: false
  boltz: false
  esm: true
  es: false
```

So the active Snakemake graph is ESM-only:

```text
Snakefile
  config.yaml
  profiles/slurm/config.yaml
  workflow/rules/esm.smk
    workflow/scripts/chunk_yamls_for_esm.py
    slurm_scripts/run_esm.py
      utils/utils.py
      utils/__init__.py
```

Because `input.yaml_dir` is commented out and both MSA and Boltz are disabled, `workflow/rules/esm.smk` uses:

```text
/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge/boltz_10k/sequences
```

as the YAML input root. That directory must already contain `.yaml` files recursively, otherwise the ESM chunking checkpoint fails with "No .yaml files found".

## Required repo files for this config

Required for Snakemake control and scheduling:

- `Snakefile`
- `config.yaml`
- `profiles/slurm/config.yaml`

Required by the active ESM stage:

- `workflow/rules/esm.smk`
- `workflow/scripts/chunk_yamls_for_esm.py`
- `slurm_scripts/run_esm.py`
- `utils/utils.py`
- `utils/__init__.py`

Required only if running through a container path:

- Container image paths under `config.containers.*`, if added later. The current `config.yaml` has no `containers:` block, so the non-container branch is used.

## Required external runtime dependencies

Snakemake launcher environment:

- `snakemake`
- `snakemake-executor-plugin-slurm`
- A Python interpreter available as `python` for local shell rules. The active local rule, `workflow/scripts/chunk_yamls_for_esm.py`, uses only the Python standard library.

Slurm execution environment:

- Slurm account: `kempner_bsabatini_lab`
- Default partition: `kempner_requeue`
- GPU access requested via `slurm_extra(gpu=True)`, which adds `--gpus-per-node=1`
- The ESM rule also attempts `module load gcc/14.2.0-fasrc01 cuda/12.9.1-fasrc01 cudnn/9.10.2.21_cuda12-fasrc01 || true`

ESM conda/env path from `config.yaml`:

- `/n/home06/tbush/envs/esm/bin/python`

Packages needed inside the ESM env:

- `esm`
- `torch`
- `numpy`
- `pandas`
- `PyYAML`
- `tqdm`

Reason: `slurm_scripts/run_esm.py` imports `esm`, `numpy`, and `utils.utils`. Importing `utils.utils` imports `pandas`, `yaml`, and `tqdm` at module import time even though the active call only uses `load_seq_`.

ESM model/cache path:

- `/n/holylfs06/LABS/bsabatini_lab/Everyone/esm_models_cache`

The ESM rule exports both `TORCH_HOME` and `HF_HOME` to that path before running `slurm_scripts/run_esm.py`.

Input/output dirs used by this config:

- Input YAML root: `/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge/boltz_10k/sequences`
- ESM chunk working dir: `/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge/boltz_10k/esm_chunks`
- ESM benchmark dir: `/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge/boltz_10k/benchmarks/esm`
- Per-sequence ESM outputs: `/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge/boltz_10k/sequences/<seq>/esm`

## Files not needed for this config

These are not needed to run the current ESM-only `snakemake --profile profiles/slurm/` path. I am intentionally not listing `viz/` here per the requested exception.

Directory-level summary:

- Fully not needed at runtime for this config: `bash_scripts/`, `tests/`, `.pytest_cache/`
- Not needed for execution, but useful for docs/setup/source control: `docs/`, `.git/`
- Partially needed: `workflow/rules/`, `workflow/scripts/`, `slurm_scripts/`, `utils/`
- Needed from partially needed dirs:
  - `workflow/rules/esm.smk`
  - `workflow/scripts/chunk_yamls_for_esm.py`
  - `slurm_scripts/run_esm.py`
  - `utils/utils.py`
  - `utils/__init__.py`

Disabled Snakemake stages:

- `workflow/rules/msa.smk`
- `workflow/rules/boltz.smk`
- `workflow/rules/es.smk`

Workflow helper scripts for disabled stages:

- `workflow/scripts/chunk_fastas.py`
- `workflow/scripts/organize_msa_outputs.py`
- `workflow/scripts/prepare_boltz_chunks.py`
- `workflow/scripts/organize_boltz_outputs.py`
- `workflow/scripts/collect_cif_paths.py`
- `workflow/scripts/run_es.py`

Legacy/manual Slurm scripts not called by the Snakemake ESM rule:

- `slurm_scripts/checker.sh`
- `slurm_scripts/checker_boltz.sh`
- `slurm_scripts/checker_esm.sh`
- `slurm_scripts/checker_msa.sh`
- `slurm_scripts/example.sh`
- `slurm_scripts/organize_boltz_outputs.sh`
- `slurm_scripts/parse_config.py`
- `slurm_scripts/process_msa_fasta.sh`
- `slurm_scripts/run_boltz_array.slrm`
- `slurm_scripts/run_boltz_organize.slrm`
- `slurm_scripts/run_boltz_wrapper.slrm`
- `slurm_scripts/run_checker_boltz.slrm`
- `slurm_scripts/run_checker_esm.slrm`
- `slurm_scripts/run_checker_msa.slrm`
- `slurm_scripts/run_es.sh`
- `slurm_scripts/run_es_array.slrm`
- `slurm_scripts/run_esm.sh`
- `slurm_scripts/run_esm_array.slrm`
- `slurm_scripts/run_esm_wrapper.slrm`
- `slurm_scripts/run_msa_array.slrm`
- `slurm_scripts/split_and_run_boltz.sh`
- `slurm_scripts/split_and_run_msa.sh`

Root-level setup and standalone wrappers not called by Snakemake:

- `setup.sh` after `config.yaml` already exists and is correct
- `download_tools.sh`
- `run.sh`
- `run_msa.sh`
- `run_boltz.sh`
- `run_boltz_predictions.sh`
- `run_esm_standalone.sh`
- `run_es_standalone.sh`

Data-generation utilities not called by Snakemake:

- `bash_scripts/generate_data.sh`
- `utils/generate_data.py`
- `utils/generate_subsamples.py`

Docs, tests, and metadata not required at runtime:

- `README.md`
- `CLAUDE.md`
- `TODO.md`
- `docs/CLUSTER_SETUP.md`
- `docs/CONTAINERS.md`
- `docs/SNAKEMAKE_GUIDE.md`
- `config.template.yaml`
- `requirements-data.txt`
- `pyproject.toml`
- `tests/`

Generated caches not needed:

- `.pytest_cache/`
- `__pycache__/` directories
- `*.pyc` files

## Files needed only for other pipeline modes

If `pipeline.msa: true`, Snakemake additionally needs:

- `workflow/rules/msa.smk`
- `workflow/scripts/chunk_fastas.py`
- `workflow/scripts/organize_msa_outputs.py`
- External `colabfold_search`, MMseqs2 database, ColabFold database/bin paths from `msa.*`

If `pipeline.boltz: true`, Snakemake additionally needs:

- `workflow/rules/boltz.smk`
- `workflow/scripts/prepare_boltz_chunks.py`
- `workflow/scripts/organize_boltz_outputs.py`
- External Boltz env/cache paths from `boltz.*`

If `pipeline.es: true`, Snakemake additionally needs:

- `workflow/rules/es.smk`
- `workflow/scripts/collect_cif_paths.py`
- `workflow/scripts/run_es.py`
- External PDAnalysis dir, ES env, and exactly one ES reference source from `es.ref_dir`, `es.ref_path`, or `es.ref_seq`

## Cleanup candidates

Safe runtime-pruning candidates for the current ESM-only Snakemake run are:

- The legacy/manual Slurm scripts listed above, except `slurm_scripts/run_esm.py`
- The standalone root wrappers listed above
- The disabled-stage workflow scripts if this repo will not be used for MSA, Boltz, or ES in the future
- `.pytest_cache/`, `__pycache__/`, and `*.pyc`

Do not remove the disabled-stage `workflow/rules/*.smk` and `workflow/scripts/*.py` files if the repo should still support toggling MSA, Boltz, or ES back on via `config.yaml`.
