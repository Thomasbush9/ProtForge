"""
Snakemake launch + process control and config I/O for the ProtForge web UI.

Pure functions (no Streamlit deps): read/write a session's config, build and
launch the snakemake command, query/stop the controller process, and validate
inputs before submitting cluster jobs.
"""

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from session import Session, touch_session, REPO_ROOT
from validate import scan_directory


def load_config(session: Session) -> dict:
    if session.config_path.exists():
        return yaml.safe_load(session.config_path.read_text()) or {}
    return {}


def save_config(session: Session, cfg: dict):
    if session.config_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session.backup_dir.mkdir(exist_ok=True)
        shutil.copy2(session.config_path, session.backup_dir / f"config_{ts}.yaml")
    session.config_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    touch_session(session.id)


def run_cmd(cmd: list[str], timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT)
        return r.stdout + r.stderr
    except FileNotFoundError:
        return f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return "Command timed out."


# Per-stage Snakemake custom-resource names (one slot per running job). A global
# budget passed via `--resources <name>=N` caps that stage to N concurrent jobs.
_STAGE_JOB_RESOURCE = {
    "msa": "msa_jobs", "boltz": "boltz_jobs", "esmc": "esmc_jobs",
    "esmfold": "esmfold_jobs", "openfold": "openfold_jobs",
}


def concurrency_resource_args(cfg: dict) -> list[str]:
    """Build `--resources <stage>_jobs=N ...` from each stage's max_concurrent_jobs.
    Stages without the key are left unbounded (default Snakemake behavior)."""
    pairs = []
    for stage, res_name in _STAGE_JOB_RESOURCE.items():
        n = cfg.get(stage, {}).get("max_concurrent_jobs")
        if n:
            pairs.append(f"{res_name}={int(n)}")
    return ["--resources", *pairs] if pairs else []


def launch_snakemake(session: Session, extra_args: list[str] | None = None):
    cfg = load_config(session)
    cmd = [
        "snakemake",
        "--profile", "profiles/slurm/",
        "--configfile", str(session.config_path),
    ]
    # Global concurrency cap (overrides the profile's `jobs:`), then per-stage caps.
    max_jobs = cfg.get("slurm", {}).get("max_concurrent_jobs")
    if max_jobs:
        cmd.extend(["--jobs", str(int(max_jobs))])
    cmd.extend(concurrency_resource_args(cfg))
    if extra_args:
        cmd.extend(extra_args)
    with open(session.log_file, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=REPO_ROOT)
    meta = {
        "pid": proc.pid,
        "session_id": session.id,
        "start_time": datetime.now().isoformat(),
        "command": " ".join(cmd),
    }
    session.run_meta_file.write_text(json.dumps(meta))
    return proc.pid


def validate_launch_inputs(cfg: dict) -> list[str]:
    """Validate configured inputs before submitting cluster jobs."""
    errors = []
    pipeline = cfg.get("pipeline", {})
    inp = cfg.get("input", {})
    if not pipeline.get("msa", False):
        # MSA off. ESMC/ESMFold2 can run straight from a FASTA (or YAML) dir;
        # Boltz/OpenFold3 can't build an MSA themselves, so they need prebuilt
        # YAMLs. Check the right input source exists for the enabled stages.
        needs_yaml = pipeline.get("boltz") or pipeline.get("openfold")
        needs_seq = pipeline.get("esmc") or pipeline.get("esmfold")
        yaml_dir_value = inp.get("yaml_dir", "")
        if needs_yaml:
            if not yaml_dir_value:
                errors.append(
                    "Boltz/OpenFold3 are enabled with MSA off, but "
                    "input.yaml_dir is not set (they need prebuilt YAMLs)."
                )
            elif not Path(yaml_dir_value).is_dir():
                errors.append(f"YAML input directory does not exist: {yaml_dir_value}")
        if needs_seq and not (inp.get("fasta_dir") or yaml_dir_value):
            errors.append(
                "ESMC/ESMFold2 are enabled with MSA off, but neither "
                "input.fasta_dir nor input.yaml_dir is set."
            )
        return errors

    fasta_dir_value = cfg.get("input", {}).get("fasta_dir", "")
    if not fasta_dir_value:
        errors.append("MSA is enabled but input.fasta_dir is not set.")
        return errors

    fasta_dir = Path(fasta_dir_value)
    if not fasta_dir.is_dir():
        errors.append(f"MSA input directory does not exist: {fasta_dir}")
        return errors

    scan_result = scan_directory(fasta_dir)
    if scan_result["file_type"] == "none":
        errors.append(f"No FASTA files found in {fasta_dir}.")
        return errors

    if scan_result["file_type"] != "fasta":
        errors.append(f"MSA input directory must contain only FASTA files: {fasta_dir}")
        return errors

    invalid_fastas = [r for r in scan_result["fasta_results"] if not r["valid"]]
    if invalid_fastas:
        first = invalid_fastas[0]
        for err in first["errors"]:
            errors.append(f"{first['filename']}: {err}")
        extra = len(invalid_fastas) - 1
        if extra > 0:
            errors.append(f"{extra} more FASTA file(s) failed validation.")

    return errors


def snakemake_status(session: Session) -> tuple[bool, int | None, str | None]:
    if not session.run_meta_file.exists():
        return False, None, None
    try:
        meta = json.loads(session.run_meta_file.read_text())
    except (json.JSONDecodeError, ValueError):
        return False, None, None
    pid = meta.get("pid")
    start_time = meta.get("start_time")
    if pid is None:
        return False, None, None
    try:
        os.kill(pid, 0)
        r = subprocess.run(["ps", "-p", str(pid), "-o", "comm="], capture_output=True, text=True, timeout=5)
        if "snakemake" not in r.stdout.lower() and "python" not in r.stdout.lower():
            session.run_meta_file.unlink(missing_ok=True)
            return False, pid, start_time
        return True, pid, start_time
    except (OSError, subprocess.TimeoutExpired):
        session.run_meta_file.unlink(missing_ok=True)
        return False, pid, start_time


def stop_snakemake(session: Session) -> bool:
    if not session.run_meta_file.exists():
        return False
    try:
        meta = json.loads(session.run_meta_file.read_text())
        pid = meta.get("pid")
        if pid:
            import signal
            os.kill(pid, signal.SIGTERM)
            session.run_meta_file.unlink(missing_ok=True)
            return True
    except (OSError, json.JSONDecodeError):
        pass
    session.run_meta_file.unlink(missing_ok=True)
    return False
