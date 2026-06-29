---
name: run-pipeline
description: >-
  Define job requirements and launch the ProtForge pipeline end to end. Use when
  the user wants to run/launch/push the pipeline, size or estimate SLURM
  resources for a run, pick a GPU/partition, prepare inputs, or submit a
  Snakemake job. Interviews the researcher, preps inputs, estimates resources
  from the calibrated scaling models, applies them to a config, dry-runs, then
  submits and monitors.
---

# Run the ProtForge pipeline

Drive a researcher from "I have sequences" to a running, correctly-sized
Snakemake job. The heavy lifting (resource estimation, GPU selection, chunk and
bin planning) already exists in `webapp/estimator.py` — this skill is the
conversational driver over it via the `webapp/estimate_cli.py` CLI plus the
normal Snakemake commands.

Do **not** reimplement estimation. Call the CLI.

## Picking the Python environment

`estimate_cli` needs only `pyyaml`; launching the pipeline needs `snakemake`.
Both live in the same env. The documented setup (see `docs/SNAKEMAKE_GUIDE.md`)
is:

```bash
module load python
mamba activate snakemake
```

Run everything below from the repo root inside that env. If `snakemake` is not
on `PATH`, ask the user how they activate it rather than guessing — envs differ
per user (e.g. `~/envs/protforge-calibrate` also has `pyyaml` for estimate-only
work).

## Procedure

### 1. Interview — pin down the job

Before touching anything, settle these. Ask only what you can't infer from an
existing config the user points at:

- **Input data** — a directory of `.fasta`/`.fa` (one sequence per file) or
  Boltz `.yaml`? Or raw CSV/mutation table that still needs converting (step 2)?
- **Stages** — which of `msa`, `boltz`, `esmc`, `esmfold`, `openfold` (and ESM-C
  `sae`)? These map to `pipeline.*` toggles. Note the dependency: Boltz/OpenFold
  consume MSA output; `esmc`/`esmfold`/`sae` run straight from sequence.
- **Smoke vs production** — recommend a smoke run first (a config pointed at a
  handful of short sequences, or `--subsample`) before committing to the full
  set. There is no checked-in example config; copy `config.template.yaml`.
- **GPU preference** — `auto` (let the estimator pick the cheapest GPU that
  fits) unless the user wants to pin one (e.g. `h100`).

### 2. Prepare inputs (only if starting from CSV / mutations)

If the user has a directory of valid FASTA/YAML already, skip this.

```bash
# name+sequence CSV -> FASTA dir
bash bash_scripts/generate_data.sh --data sequences.csv
# mutation table -> FASTA dir (needs a reference)
bash bash_scripts/generate_data.sh --data mutations.tsv --original reference.fasta
# add --file_type yaml to emit Boltz YAMLs (skips MSA)
```

The script prints the output dir — that becomes `input.fasta_dir` (or
`input.yaml_dir`) in the config.

### 3. Get a config to work with

- Reuse an existing config the user names, **or**
- `cp config.template.yaml config.<run>.yaml` and fill in the must-haves:
  `pipeline.*` toggles, `input.fasta_dir`/`input.yaml_dir`,
  `output.parent_dir`, `slurm.account`, `slurm.log_dir`, container `.sif` paths,
  and shared DB/cache paths (`msa.mmseq2_db`, `boltz.cache_dir`,
  `esmc.cache_dir`). See the template comments and `docs/CLUSTER_SETUP.md`.

Resources (mem/runtime/cpus/chunks) are **not** filled by hand — the estimator
writes them in step 5.

### 4. Estimate resources (read-only)

```bash
python -m webapp.estimate_cli --config config.<run>.yaml
```

This scans the input dir (precedence `input.yaml_dir` > `input.fasta_dir`, or
pass `--input-dir`), summarizes sequence lengths, and prints a per-stage table:
GPU, partition, mem, runtime, cpus, chunk size, chunk count, node-hours.

Useful flags:
- `--gpu h100` — pin a GPU for every stage; `--gpu boltz=h100` — pin one stage
  (repeatable, mix bare + per-stage; `auto` = let it pick).
- `--json` — machine-readable output if you need to parse it.

Read the **Notes** section out to the user. In particular:
- An OOM-risk note on a real GPU stage (boltz/esmc/esmfold) means the data is too
  big for that GPU — suggest pinning a bigger one or trimming long sequences
  (`boltz.max_seq_len`).
- An MSA "exceeds largest known GPU" note is expected and **not** a real GPU
  OOM: MSA's large memory is host RAM for the mmap'd ColabFold DB (~256 GB),
  which the estimator requests as `--mem`. Reassure rather than alarm.
- **The MSA mem estimate over-requests** (e.g. ~325 GB) — actual MSA memory is
  roughly **fixed** (the DB mmap footprint, ~135 GB observed) and does not scale
  the way the estimate implies. After `--apply`, cap it by hand:
  `slurm.resources.msa.mem_mb: 256000`. 256 GB leaves ample headroom and avoids
  wasteful requests that can hurt queue time on constrained partitions.

Caveat: the estimator covers `msa`, `boltz`, `esmc`, `esmfold` only. It does
**not** size `openfold`, per-size ESM-C (`esmc_6B`/`600M`/`300M`), or `sae`
jobs — set those `slurm.resources.*` entries by hand if the run uses them.

### 5. Apply to the config

Preview first, then write (a timestamped backup lands in
`<config_dir>/.config_backups/`):

```bash
python -m webapp.estimate_cli --config config.<run>.yaml --apply --dry-run  # preview
python -m webapp.estimate_cli --config config.<run>.yaml --apply            # write
```

This sets `slurm.resources.<stage>`, the stage partition, and
`<stage>.max_files_per_job` (and a `binning` recipe if enabled). It preserves
all other keys.

### 6. Dry-run the DAG

```bash
snakemake --profile profiles/slurm/ --configfile config.<run>.yaml -n
```

Sanity-check the job/stage counts against the estimate before submitting. If
something looks off, fix the config and re-dry-run — don't submit on a surprise.

### 7. Launch

```bash
snakemake --profile profiles/slurm/ --configfile config.<run>.yaml
# cap global concurrency if asked, e.g. --jobs 10
```

### 8. Monitor

```bash
squeue -u $USER                                                  # live jobs
snakemake --profile profiles/slurm/ --configfile config.<run>.yaml --summary
# resume after a failure/preemption (picks up where it left off):
snakemake --profile profiles/slurm/ --configfile config.<run>.yaml --rerun-incomplete
```

On failures, read the SLURM logs under `slurm.log_dir` and report the actual
error; suggest `--rerun-incomplete` for transient/preemption failures.

**Known failure mode — Boltz `cudaErrorDevicesUnavailable`.** If a Boltz (or
other GPU) job dies with `torch.AcceleratorError: CUDA error: CUDA-capable
device(s) is/are busy or unavailable` at `torch.cuda.set_device()`, that is
**transient GPU contention on a shared partition** (the estimator often places
Boltz on shared `kempner`/a100), NOT a config/data error — chunks that got clean
GPUs finish fine. Recovery: set `slurm.boltz.partition: kempner_h100` (dedicated,
less contended) and resume with `--rerun-incomplete`. The resume redoes only the
failed chunks + downstream organize/sentinel steps; completed stages are kept.
Note the rule's `"no Boltz container configured"` guard text appears in the log
dump even when the container IS set — read the Python traceback for the real
cause, not that line.

**`mamba` won't be on `PATH` across separate Bash calls** (shell state doesn't
persist, and `mamba` is a shell function). Chain in one command:
`module load python && mamba activate snakemake && <cmd>`. The long-running
orchestrator should be launched in the background (`nohup … &`) so it can keep
submitting downstream jobs as checkpoints complete.

## Notes

- Calibration data (`scaling_models.yaml`, v4 H100/A100 fits) lives in `webapp/`.
  An optional `webapp/scaling_models.calibrated.yaml` overlay shadows it if
  present. (Re)fitting it is the separate stair-sweep workflow under
  `scripts/calibrate/stair/` — out of scope for this skill.
- Lab notes (decisions, calibration findings, run IDs) live in the vault under
  `~/Documents/Vault/Notes/Lab/protforge/`, not in the repo.
