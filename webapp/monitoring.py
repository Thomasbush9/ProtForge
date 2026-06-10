"""
Cluster + output monitoring helpers for the ProtForge web UI.

Pure functions (no Streamlit deps): SLURM queries via squeue/sacct, output
artifact counting for per-stage progress, and small formatting helpers. The
Job Monitor tab renders on top of these.
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from session import REPO_ROOT

USER = os.environ.get("USER", "unknown")

# Per-stage auto-refresh intervals (seconds)
REFRESH_INTERVALS = {"MSA": 300, "Boltz": 60, "ESMC": 30, "ESMC-SAE": 30,
                     "ESMFold": 60, "OpenFold": 60}

# Map Snakemake rule names to pipeline stages. ESMC/ESMC-SAE rules carry a
# {size} wildcard, so match on the rule-name prefix (any rule containing the key).
RULE_TO_STAGE = {
    "run_colabfold_search": "MSA",
    "scatter_msa_and_create_yaml": "MSA",
    "chunk_fastas": "MSA",
    "msa_complete": "MSA",
    "run_boltz_predict": "Boltz",
    "organize_boltz_chunk": "Boltz",
    "chunk_yamls_for_boltz": "Boltz",
    "boltz_complete": "Boltz",
    # SAE keys first so they win over the "esmc" substring match below.
    "run_esmc_sae": "ESMC-SAE",
    "chunk_yamls_for_sae": "ESMC-SAE",
    "organize_esmc_sae": "ESMC-SAE",
    "esmc_sae_complete": "ESMC-SAE",
    "run_esmc": "ESMC",
    "chunk_yamls_for_esmc": "ESMC",
    "organize_esmc": "ESMC",
    "esmc_complete": "ESMC",
    "run_esmfold": "ESMFold",
    "chunk_yamls_for_esmfold": "ESMFold",
    "organize_esmfold": "ESMFold",
    "esmfold_complete": "ESMFold",
    "run_openfold_predict": "OpenFold",
    "chunk_yamls_for_openfold": "OpenFold",
    "organize_openfold_chunk": "OpenFold",
    "openfold_complete": "OpenFold",
}


def format_elapsed(start_iso: str) -> str:
    start = datetime.fromisoformat(start_iso)
    delta = datetime.now() - start
    total_secs = int(delta.total_seconds())
    hours, remainder = divmod(total_secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def get_slurm_jobs() -> list[dict]:
    try:
        r = subprocess.run(
            ["squeue", "-u", USER, "--json"],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(r.stdout)
        return data.get("jobs", [])
    except Exception:
        return []


def get_recent_jobs(hours: int = 24) -> list[dict]:
    try:
        r = subprocess.run(
            ["sacct", "-u", USER, "--starttime", f"now-{hours}hours",
             "--format=JobID,JobName%50,State,ExitCode,Elapsed,Start,End,Reason%30",
             "--noheader", "--parsable2"],
            capture_output=True, text=True, timeout=15,
        )
        jobs = []
        for line in r.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 8:
                continue
            job_id = parts[0]
            if "." in job_id:
                continue
            jobs.append({
                "job_id": job_id,
                "name": parts[1].strip(),
                "state": parts[2],
                "exit_code": parts[3],
                "elapsed": parts[4],
                "start": parts[5],
                "end": parts[6],
                "reason": parts[7].strip(),
            })
        return jobs
    except Exception:
        return []


def is_protforge_job(job_name: str) -> bool:
    return any(rule in job_name for rule in RULE_TO_STAGE)


def job_to_stage(job_name: str) -> str:
    for rule, stage in RULE_TO_STAGE.items():
        if rule in job_name:
            return stage
    return "Other"


def get_job_log_path(job_id: int | str, cfg: dict) -> Path | None:
    slurm_logs = REPO_ROOT / ".snakemake" / "slurm_logs"
    if slurm_logs.exists():
        for log in slurm_logs.rglob(f"{job_id}.log"):
            return log
    log_dir = cfg.get("slurm", {}).get("log_dir", "")
    if log_dir:
        log_path = Path(log_dir)
        if log_path.exists():
            for pattern in [f"*{job_id}*.out", f"*{job_id}*.log", f"*{job_id}*.err"]:
                matches = list(log_path.glob(pattern))
                if matches:
                    return matches[0]
    return None


def read_log_tail(path: Path, n_lines: int = 100) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception as e:
        return f"Error reading log: {e}"


def count_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def get_expected_total(cfg: dict) -> int:
    output_dir = Path(cfg.get("output", {}).get("parent_dir", ""))
    seq_dir = output_dir / "sequences"
    fasta_dir = Path(cfg.get("input", {}).get("fasta_dir", ""))
    yaml_dir = Path(cfg.get("input", {}).get("yaml_dir", ""))
    pipeline = cfg.get("pipeline", {})

    if not pipeline.get("msa") and yaml_dir != Path("") and yaml_dir.is_dir():
        return count_files(yaml_dir, "*.yaml")
    if fasta_dir.is_dir():
        return count_files(fasta_dir, "*.fasta") + count_files(fasta_dir, "*.fa")
    if seq_dir.is_dir():
        return len([d for d in seq_dir.iterdir() if d.is_dir()])
    return 0


def get_stage_progress(cfg: dict) -> dict:
    output_dir = Path(cfg.get("output", {}).get("parent_dir", ""))
    seq_dir = output_dir / "sequences"
    pipeline = cfg.get("pipeline", {})
    num_runs = cfg.get("boltz", {}).get("num_runs", 1)

    total = get_expected_total(cfg)
    progress = {}

    if pipeline.get("msa"):
        # Count sequence dirs that have an MSA, not a3m files: the OpenFold
        # converter adds a second a3m (uniref90_hits.a3m) per msa/, so a file
        # count would exceed `total` and overflow the progress bar.
        done = 0
        if seq_dir.is_dir():
            for d in seq_dir.iterdir():
                msa_d = d / "msa"
                if msa_d.is_dir() and any(msa_d.glob("*.a3m")):
                    done += 1
        progress["MSA"] = (done, total)

    if pipeline.get("boltz"):
        if num_runs <= 1:
            done = 0
            if seq_dir.is_dir():
                for d in seq_dir.iterdir():
                    boltz_dir = d / "boltz"
                    if boltz_dir.is_dir() and list(boltz_dir.glob("*_model_*.cif")):
                        done += 1
            progress["Boltz"] = (done, total)
        else:
            done = 0
            if seq_dir.is_dir():
                for d in seq_dir.iterdir():
                    boltz_dir = d / "boltz"
                    if boltz_dir.is_dir():
                        for run_dir in boltz_dir.glob("run_*"):
                            if list(run_dir.glob("*_model_*.cif")):
                                done += 1
            progress["Boltz"] = (done, total * num_runs)

    if pipeline.get("esmc"):
        # One outputs.pt per (sequence, model size): sequences/{seq}/esmc/{size}/outputs.pt
        sizes = cfg.get("esmc", {}).get("models", []) or []
        done = count_files(seq_dir, "*/esmc/*/outputs.pt")
        progress["ESMC"] = (done, total * max(len(sizes), 1))

    sae_cfg = cfg.get("esmc", {}).get("sae", {})
    if sae_cfg.get("enabled"):
        # One sae/{size}/{sae_type}/ dir per (sequence, size); count non-empty ones.
        sae_sizes = sae_cfg.get("sizes") or cfg.get("esmc", {}).get("models", []) or []
        done = 0
        if seq_dir.is_dir():
            for d in seq_dir.iterdir():
                sae_root = d / "sae"
                if sae_root.is_dir():
                    for size_dir in sae_root.iterdir():
                        if size_dir.is_dir() and list(size_dir.glob("*/sae_*.pt")):
                            done += 1
        progress["ESMC-SAE"] = (done, total * max(len(sae_sizes), 1))

    if pipeline.get("esmfold"):
        # sequences/{seq}/esmfold/fast/structure.cif
        done = count_files(seq_dir, "*/esmfold/*/structure.cif")
        progress["ESMFold"] = (done, total)

    if pipeline.get("openfold"):
        # One kept model per sequence: sequences/{seq}/openfold/*_model.cif
        done = 0
        if seq_dir.is_dir():
            for d in seq_dir.iterdir():
                of_dir = d / "openfold"
                if of_dir.is_dir() and any(of_dir.glob("*_model.*")):
                    done += 1
        progress["OpenFold"] = (done, total)

    return progress


def fastest_refresh(progress: dict) -> int | None:
    intervals = []
    for stage, (done, total) in progress.items():
        if total > 0 and done < total:
            intervals.append(REFRESH_INTERVALS.get(stage, 60))
    return min(intervals) if intervals else None
