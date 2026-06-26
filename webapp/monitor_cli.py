"""
Command-line front door to the ProtForge job monitor.

Wraps the same pure functions the Streamlit Job Monitor tab uses
(webapp/monitoring.py) so a run can be watched and triaged without the web app:

    python -m webapp.monitor_cli --config config.ga.yaml             # progress + jobs
    python -m webapp.monitor_cli --config config.ga.yaml --json       # machine-readable
    python -m webapp.monitor_cli --config config.ga.yaml --recent 48  # job history, last 48h
    python -m webapp.monitor_cli --config config.ga.yaml --failed     # only failed/cancelled
    python -m webapp.monitor_cli --config config.ga.yaml --log 12345  # tail a job's log

Flow: read config -> per-stage pipeline progress (done/total from the config's
output dir) -> current SLURM jobs for the user (squeue) -> recent/failed jobs
(sacct) -> optionally tail a single job's SLURM log.

Designed to be driven by the `monitor` Claude Code skill, but usable standalone.
Read-only: it never submits, cancels, or modifies anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

# The webapp modules use flat imports (Streamlit runs from inside webapp/).
# Make that work whether this is invoked as `python -m webapp.monitor_cli`
# or `python webapp/monitor_cli.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monitoring import (  # noqa: E402
    get_slurm_jobs,
    get_recent_jobs,
    get_stage_progress,
    get_job_log_path,
    read_log_tail,
    is_protforge_job,
    job_to_stage,
)

# States we treat as "failed" for the --failed filter / triage view.
FAILED_STATES = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                 "BOOT_FAIL", "DEADLINE", "PREEMPTED")


def _job_state(job: dict) -> str:
    """squeue --json gives job_state as a list; normalize to a string."""
    state = job.get("job_state", job.get("state", "UNKNOWN"))
    if isinstance(state, list):
        state = state[0] if state else "UNKNOWN"
    return str(state)


def _is_failed(state: str) -> bool:
    return any(state.upper().startswith(s) for s in FAILED_STATES)


def _print_table(header: list[str], rows: list[list[str]]) -> None:
    """ASCII table printer matching estimate_cli's style."""
    if not rows:
        return
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("  ".join("-" * widths[i] for i in range(len(header))))
    for r in rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))


def _print_progress(progress: dict) -> None:
    print("\nPipeline progress (done / total from output dir):")
    if not progress:
        print("  No stages enabled, or output.parent_dir not set / missing.")
        return
    header = ["stage", "done", "total", "pct", "bar"]
    rows = []
    for stage, (done, total) in progress.items():
        frac = (done / total) if total > 0 else 0.0
        frac = max(0.0, min(1.0, frac))
        filled = int(round(frac * 20))
        bar = "#" * filled + "." * (20 - filled)
        rows.append([stage, str(done), str(total), f"{frac * 100:.0f}%", bar])
    _print_table(header, rows)


def _print_active(jobs: list[dict], show_all: bool) -> None:
    pf = [j for j in jobs if is_protforge_job(j.get("name", ""))]
    shown = jobs if show_all else pf
    print("\nActive SLURM jobs"
          f" ({'all user jobs' if show_all else 'ProtForge only'}):")
    if not shown:
        if jobs and not pf:
            print(f"  No ProtForge jobs running ({len(jobs)} other user jobs active).")
        else:
            print("  No SLURM jobs running (or squeue returned nothing here).")
        return
    header = ["job_id", "stage", "state", "partition", "rule"]
    rows = []
    for j in shown:
        name = j.get("name", "")
        rows.append([
            str(j.get("job_id", "")),
            job_to_stage(name),
            _job_state(j),
            str(j.get("partition", "") or "-"),
            name or "-",
        ])
    _print_table(header, rows)


def _print_recent(recent: list[dict], show_all: bool, failed_only: bool) -> None:
    pf = [j for j in recent if is_protforge_job(j.get("name", ""))]
    shown = recent if show_all else pf
    if failed_only:
        shown = [j for j in shown if _is_failed(j.get("state", ""))]
    label = "failed/cancelled" if failed_only else "recent"
    scope = "all user jobs" if show_all else "ProtForge only"
    print(f"\n{label.capitalize()} job history ({scope}):")
    if not shown:
        print("  None found (sacct returned nothing here, or none match).")
        return
    header = ["job_id", "stage", "state", "exit", "elapsed", "reason"]
    rows = []
    for j in shown:
        name = j.get("name", "")
        rows.append([
            str(j.get("job_id", "")),
            job_to_stage(name),
            j.get("state", "?"),
            j.get("exit_code", "-"),
            j.get("elapsed", "-"),
            (j.get("reason", "") or "-")[:30],
        ])
    _print_table(header, rows)
    if failed_only:
        print("\nTail a failing job's log with:  --log <job_id>")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="monitor_cli",
        description="Monitor ProtForge job status, triage failures (read-only).",
    )
    p.add_argument("--config", required=True, help="Path to the pipeline config YAML.")
    p.add_argument("--all", action="store_true",
                   help="Show all of the user's SLURM jobs, not just ProtForge ones.")
    p.add_argument("--recent", type=int, metavar="HOURS", nargs="?", const=24,
                   help="Show job history from the last HOURS (default 24).")
    p.add_argument("--failed", action="store_true",
                   help="Only show failed/cancelled jobs in the history view.")
    p.add_argument("--log", metavar="JOBID",
                   help="Tail the SLURM log for one job id and exit.")
    p.add_argument("--lines", type=int, default=150,
                   help="Lines to show with --log (default 150).")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of the tables.")
    args = p.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    # --log: single-purpose, tail one job's log and stop.
    if args.log:
        log_path = get_job_log_path(args.log, cfg)
        if not log_path:
            log_dir = cfg.get("slurm", {}).get("log_dir", "(slurm.log_dir not set)")
            raise SystemExit(
                f"No log file found for job {args.log}. Looked under "
                f".snakemake/slurm_logs/ and {log_dir}."
            )
        if args.json:
            print(json.dumps({
                "job_id": args.log,
                "log_path": str(log_path),
                "tail": read_log_tail(log_path, args.lines),
            }, indent=2))
        else:
            print(f"Log: {log_path}\n")
            print(read_log_tail(log_path, args.lines))
        return 0

    # When --failed is asked without --recent, default the history window.
    recent_hours = args.recent if args.recent is not None else (24 if args.failed else None)

    progress = get_stage_progress(cfg)
    active = get_slurm_jobs()
    recent = get_recent_jobs(hours=recent_hours) if recent_hours is not None else []

    if args.json:
        payload = {
            "config": str(config_path),
            "output_dir": cfg.get("output", {}).get("parent_dir", ""),
            "progress": {s: {"done": d, "total": t} for s, (d, t) in progress.items()},
            "active_jobs": [
                {**j, "_stage": job_to_stage(j.get("name", "")),
                 "_state": _job_state(j),
                 "_is_protforge": is_protforge_job(j.get("name", ""))}
                for j in active
            ],
            "recent_jobs": [
                {**j, "_stage": job_to_stage(j.get("name", "")),
                 "_is_protforge": is_protforge_job(j.get("name", "")),
                 "_is_failed": _is_failed(j.get("state", ""))}
                for j in recent
            ],
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(f"Config: {config_path}")
    out = cfg.get("output", {}).get("parent_dir", "")
    print(f"Output dir: {out or '(output.parent_dir not set)'}")
    if os.environ.get("USER"):
        print(f"User: {os.environ['USER']}")

    _print_progress(progress)
    _print_active(active, show_all=args.all)
    if recent_hours is not None:
        print(f"\n(history window: last {recent_hours}h)")
        _print_recent(recent, show_all=args.all, failed_only=args.failed)
    else:
        print("\nAdd --recent [HOURS] for job history, or --failed for failures only.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
