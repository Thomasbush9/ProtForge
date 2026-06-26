---
name: monitor
description: >-
  Check ProtForge job status, triage failures, and recover a run. Use when the
  user asks to check job/pipeline status, see how far a run has gotten, ask "why
  did my run fail" / "what's stuck", monitor the pipeline, inspect a job's log,
  or recover/resume failed jobs. Drives the read-only webapp/monitor_cli.py over
  squeue/sacct plus output-artifact counting, then points at recovery.
---

# Monitor the ProtForge pipeline

Tell a researcher where their run stands and what to do about failures. The
queries (SLURM jobs, sacct history, per-stage progress from output artifacts,
log tailing) already exist in `webapp/monitoring.py` — this skill is the driver
over them via the `webapp/monitor_cli.py` CLI. It is **read-only**: it never
submits, cancels, or edits anything.

Do **not** reimplement monitoring. Call the CLI.

## Picking the Python environment

`monitor_cli` needs only `pyyaml`. Run from the repo root. The documented setup
(see `docs/SNAKEMAKE_GUIDE.md`) is `module load python && mamba activate
snakemake`; the calibrate env also works:

```bash
~/envs/protforge-calibrate/bin/python -m webapp.monitor_cli --config <cfg>
```

If `python` isn't on `PATH`, use that interpreter or ask how the user activates
their env rather than guessing.

## Commands

Point `--config` at the same config the run used (its `output.parent_dir` is
where progress is counted; `slurm.log_dir` is where logs are found).

```bash
# Overview: per-stage progress + active SLURM jobs
python -m webapp.monitor_cli --config config.<run>.yaml

# Add job history (sacct), last N hours (default 24)
python -m webapp.monitor_cli --config config.<run>.yaml --recent 48

# Only failed / cancelled / timed-out / OOM jobs
python -m webapp.monitor_cli --config config.<run>.yaml --failed

# Tail one job's SLURM log (the actual error lives here)
python -m webapp.monitor_cli --config config.<run>.yaml --log <jobid> --lines 200

# Include non-ProtForge jobs, or get machine-readable output
python -m webapp.monitor_cli --config config.<run>.yaml --all
python -m webapp.monitor_cli --config config.<run>.yaml --json
```

Notes on the environment: `squeue`/`sacct` may return nothing (e.g. on a login
shell with no jobs) — the CLI says so rather than crashing. By default only
ProtForge jobs are shown (matched by Snakemake rule name); `--all` widens it.

## Reading failures

1. Run `--failed` (or `--recent`) to find the failing job id, stage, and the
   sacct `state`/`reason` (e.g. `OUT_OF_MEMORY`, `TIMEOUT`, `PREEMPTED`).
2. Run `--log <jobid>` to read the real error from the job's SLURM log. Logs
   live under `.snakemake/slurm_logs/` and the config's `slurm.log_dir`.
3. Interpret and report the actual error, not just the state:
   - `OUT_OF_MEMORY` / OOM in the log → resize that stage (re-estimate with the
     `run-pipeline` skill / `estimate_cli`, e.g. pin a bigger GPU or trim long
     sequences) before resubmitting.
   - `TIMEOUT` → bump the stage's runtime (re-estimate) or shrink chunks.
   - `PREEMPTED` / `NODE_FAIL` → transient; just resume (below).
   - A real code/data error in the log → fix the input/config; resuming alone
     won't help.

## Recovery

For transient/preemption failures, resume — Snakemake picks up where it left
off and only reruns incomplete work:

```bash
snakemake --profile profiles/slurm/ --configfile config.<run>.yaml --rerun-incomplete
```

Re-run `monitor_cli` afterward to confirm the stage progress climbs and the
formerly-failed jobs complete. For resource-caused failures (OOM/timeout),
re-estimate and apply new resources first (see the `run-pipeline` skill), then
`--rerun-incomplete`.

## Notes

- Lab notes (run IDs, decisions, calibration findings) live in the vault under
  `~/Documents/Vault/Notes/Lab/protforge/`, not the repo.
