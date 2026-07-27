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

# Importing session put workflow/scripts on sys.path. Reuse the workflow's own
# expansion and identity checks so preflight fails on exactly what the Snakefile
# would reject, rather than drifting into a second set of rules.
from snake_helpers import expand_config, slurm_identity_errors  # noqa: E402


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


def check_unexpanded_paths(cfg: dict) -> list[str]:
    """Fatal `${VAR}` placeholders that the workflow would refuse to expand.

    The webapp seeds a session from the shipped template without requiring
    PROTFORGE_ROOT, so this is where an un-exported variable surfaces — in the
    UI at preflight, rather than as a traceback in the snakemake log.
    """
    try:
        expand_config(cfg)
    except KeyError as exc:
        return [str(exc).strip('"')]
    return []


# Enabled stage -> (containers.<key>, weight-cache config path). The MSA
# databases and the Boltz checkpoint are shared cluster paths, so only the
# per-user HF/OpenFold caches are checked here.
_STAGE_ASSETS = {
    "msa": ("colabfold", None),
    "boltz": ("boltz", None),
    "esmc": ("esmc", ("esmc", "cache_dir")),
    "esmfold": ("esmfold", ("esmfold", "cache_dir")),
    "openfold": ("openfold", ("openfold", "cache_dir")),
}


def check_stage_assets(cfg: dict) -> list[str]:
    """Fatal missing container images or model caches for the enabled stages.

    A seeded config points at `$PROTFORGE_ROOT/sifs/*.sif` and
    `$PROTFORGE_ROOT/models/hf`, which only exist once containers/build.sh and
    scripts/download_models.py have been run. Without this the config validates,
    every job is submitted, and each one dies on a missing image.
    """
    errors: list[str] = []
    pipeline = cfg.get("pipeline", {}) or {}
    containers = cfg.get("containers", {}) or {}
    shared_gpu = containers.get("gpu", "")

    for stage, (container_key, cache_key) in _STAGE_ASSETS.items():
        if not pipeline.get(stage):
            continue

        sif = containers.get(container_key, "") or shared_gpu
        if not sif:
            errors.append(
                f"{stage} is enabled but containers.{container_key} is not set."
            )
        elif not Path(sif).exists():
            errors.append(
                f"{stage} container image not found: {sif} — build it with "
                f"`bash containers/build.sh` (see docs/CLUSTER_SETUP.md)."
            )

        if cache_key:
            cache = cfg.get(cache_key[0], {}).get(cache_key[1], "")
            if not cache:
                errors.append(
                    f"{stage} is enabled but {cache_key[0]}.{cache_key[1]} is not set."
                )
            elif not Path(cache).is_dir():
                errors.append(
                    f"{stage} model cache not found: {cache} — download weights "
                    f"with `python scripts/download_models.py --cache-dir {cache}`."
                )

    return errors


def check_launch_inputs(cfg: dict) -> dict:
    """Inspect configured inputs before submitting cluster jobs.

    Distinguishes two kinds of problem:

    * Fatal errors (``errors``) — missing config, a missing input directory, or
      a directory of the wrong type. These block the launch outright.
    * Skippable per-file failures (``invalid_groups``) — individual FASTA/YAML
      files that fail validation. These do NOT have to block the run: the
      offending files can be excluded (renamed with ``.invalid`` so the
      chunkers skip them) and the valid remainder processed. Each group is
      ``{"dir", "scan_result", "invalid": [...], "valid_count"}`` where
      ``scan_result`` is the value to hand to ``validate.exclude_invalid_files``.

    Returns ``{"errors": [...], "invalid_groups": [...]}``.
    """
    # Identity and placeholder checks run first and ride along on every return
    # path below, several of which bail out early on an input problem.
    errors: list[str] = (
        slurm_identity_errors(cfg)
        + check_unexpanded_paths(cfg)
        + check_stage_assets(cfg)
    )
    invalid_groups: list[dict] = []
    pipeline = cfg.get("pipeline", {})
    inp = cfg.get("input", {})

    def _collect_invalid(dir_value: str):
        scan = scan_directory(Path(dir_value))
        invalid = [
            r for r in (scan["fasta_results"] + scan["yaml_results"]) if not r["valid"]
        ]
        if invalid:
            invalid_groups.append({
                "dir": dir_value,
                "scan_result": scan,
                "invalid": invalid,
                "valid_count": scan["valid_count"],
            })
        return scan

    if not pipeline.get("msa", False):
        # MSA off. ESMC/ESMFold2 can run straight from a FASTA (or YAML) dir;
        # Boltz/OpenFold3 can't build an MSA themselves, so they need prebuilt
        # YAMLs. Check the right input source exists for the enabled stages.
        needs_yaml = pipeline.get("boltz") or pipeline.get("openfold")
        needs_seq = pipeline.get("esmc") or pipeline.get("esmfold")
        yaml_dir_value = inp.get("yaml_dir", "")
        fasta_dir_value = inp.get("fasta_dir", "")
        scanned: set[str] = set()
        if needs_yaml:
            if not yaml_dir_value:
                errors.append(
                    "Boltz/OpenFold3 are enabled with MSA off, but "
                    "input.yaml_dir is not set (they need prebuilt YAMLs)."
                )
            elif not Path(yaml_dir_value).is_dir():
                errors.append(f"YAML input directory does not exist: {yaml_dir_value}")
            else:
                _collect_invalid(yaml_dir_value)
                scanned.add(yaml_dir_value)
        if needs_seq:
            # esmc/esmfold precedence: explicit yaml_dir, else the FASTA dir.
            seq_src = yaml_dir_value or fasta_dir_value
            if not seq_src:
                errors.append(
                    "ESMC/ESMFold2 are enabled with MSA off, but neither "
                    "input.fasta_dir nor input.yaml_dir is set."
                )
            elif not Path(seq_src).is_dir():
                errors.append(f"Input directory does not exist: {seq_src}")
            elif seq_src not in scanned:
                _collect_invalid(seq_src)
        return {"errors": errors, "invalid_groups": invalid_groups}

    fasta_dir_value = inp.get("fasta_dir", "")
    if not fasta_dir_value:
        errors.append("MSA is enabled but input.fasta_dir is not set.")
        return {"errors": errors, "invalid_groups": invalid_groups}

    fasta_dir = Path(fasta_dir_value)
    if not fasta_dir.is_dir():
        errors.append(f"MSA input directory does not exist: {fasta_dir}")
        return {"errors": errors, "invalid_groups": invalid_groups}

    scan_result = scan_directory(fasta_dir)
    if scan_result["file_type"] == "none":
        errors.append(f"No FASTA files found in {fasta_dir}.")
        return {"errors": errors, "invalid_groups": invalid_groups}

    if scan_result["file_type"] != "fasta":
        errors.append(f"MSA input directory must contain only FASTA files: {fasta_dir}")
        return {"errors": errors, "invalid_groups": invalid_groups}

    invalid_fastas = [r for r in scan_result["fasta_results"] if not r["valid"]]
    if invalid_fastas:
        invalid_groups.append({
            "dir": fasta_dir_value,
            "scan_result": scan_result,
            "invalid": invalid_fastas,
            "valid_count": scan_result["valid_count"],
        })
    if scan_result["valid_count"] == 0:
        # Every file failed — excluding them would leave nothing to run.
        errors.append(f"No valid FASTA files in {fasta_dir} (all failed validation).")

    return {"errors": errors, "invalid_groups": invalid_groups}


def validate_launch_inputs(cfg: dict) -> list[str]:
    """Fatal launch blockers only — back-compat wrapper over check_launch_inputs."""
    return check_launch_inputs(cfg)["errors"]


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
