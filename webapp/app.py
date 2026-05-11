"""
ProtForge Web UI — multi-session support.

Usage (on cluster login node):
    streamlit run webapp/app.py --server.port 8501 --server.address 127.0.0.1

Then on your laptop:
    ssh -L 8501:localhost:8501 <user>@<cluster>

Open http://localhost:8501 in your browser.
"""

import json
import os
import shutil
import subprocess
import socket
from datetime import datetime
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yaml

from validate import scan_directory, copy_valid_files
from estimator import (
    ALL_STAGES,
    apply_estimate_to_config,
    compute_input_stats,
    estimate_all_stages,
    load_scaling_models,
)
from session import (
    Session,
    migrate_legacy,
    load_registry,
    save_registry,
    create_session,
    delete_session,
    rename_session,
    touch_session,
    list_sessions,
    get_session,
    get_active_session_id,
    set_active_session,
    REPO_ROOT,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
USER = os.environ.get("USER", "unknown")
HOST = socket.gethostname()

# Per-stage auto-refresh intervals (seconds)
REFRESH_INTERVALS = {"MSA": 300, "Boltz": 60, "ESM": 10, "ESMFold": 60, "ES": 10}

# Map Snakemake rule names to pipeline stages
RULE_TO_STAGE = {
    "run_colabfold_search": "MSA",
    "scatter_msa_and_create_yaml": "MSA",
    "chunk_fastas": "MSA",
    "msa_complete": "MSA",
    "run_boltz_predict": "Boltz",
    "organize_boltz_chunk": "Boltz",
    "chunk_yamls_for_boltz": "Boltz",
    "boltz_complete": "Boltz",
    "run_esm_chunk": "ESM",
    "chunk_yamls_for_esm": "ESM",
    "esm_complete": "ESM",
    "run_esmfold_chunk": "ESMFold",
    "chunk_yamls_for_esmfold": "ESMFold",
    "esmfold_complete": "ESMFold",
    "collect_es_paths": "ES",
    "run_es_all": "ES",
}

# ---------------------------------------------------------------------------
# Helpers — all session-aware
# ---------------------------------------------------------------------------

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


def launch_snakemake(session: Session, extra_args: list[str] | None = None):
    cmd = [
        "snakemake",
        "--profile", "profiles/slurm/",
        "--configfile", str(session.config_path),
    ]
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
    if not pipeline.get("msa", False):
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


@st.cache_data(show_spinner="Scanning directory…")
def _cached_scan(path_str: str, dir_mtime: float) -> dict:
    """scan_directory wrapped with a cache key that invalidates when the dir
    is touched. mtime alone is good enough for the common case (adding or
    removing files); the explicit Re-scan button below covers content edits."""
    return scan_directory(Path(path_str))


def autoscan_directory(path_str: str) -> dict | None:
    """Return a scan_directory() result for `path_str`, or None if it's not
    a usable directory. Cached so typing in the path field stays snappy."""
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_dir():
        return None
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    return _cached_scan(path_str, mtime)


def render_gpu_preference(stage: str) -> None:
    """Per-stage GPU dropdown that persists in st.session_state["gpu_preferences"]."""
    if stage not in {"msa", "boltz", "esm", "esmfold"}:
        return
    prefs = st.session_state.setdefault("gpu_preferences", {})
    options = ["auto", "a100", "h100"]
    current = prefs.get(stage, "auto")
    if current not in options:
        current = "auto"
    chosen = st.selectbox(
        "GPU preference",
        options=options,
        index=options.index(current),
        key=f"gpu_pref_{stage}",
        help="auto = pick the cheapest GPU whose memory ceiling covers the "
             "estimated need. Pin a card if you have a specific requirement.",
    )
    prefs[stage] = chosen


def render_chunk_recommendation(stage: str) -> None:
    """If a recent estimate is cached, show 'Recommended: N' for this stage."""
    est = st.session_state.get("last_estimate", {}).get(stage)
    if not est:
        return
    if stage in {"msa", "boltz"}:
        st.caption(
            f"Recommended chunk size: **{est['chunk_size']}** "
            f"({est['num_chunks']} jobs, ~{est['runtime_min']} min/job)"
        )
    elif stage in {"esm", "esmfold"}:
        st.caption(
            f"Recommended num_chunks: **{est['num_chunks']}** "
            f"({est['chunk_size']} seqs/chunk, ~{est['runtime_min']} min/chunk)"
        )


# Per-stage SLURM resource defaults — kept in sync with the rule fallbacks so a
# blank session picks up reasonable values without forcing the user to run the
# estimator first. Tuples are (mem_mb, runtime_min, cpus_per_task).
_SLURM_DEFAULTS: dict[str, tuple[int, int, int]] = {
    "msa":     (256000,  60, 4),
    "boltz":   ( 16000,  60, 8),
    "esm":     ( 32000,  60, 16),
    "esmfold": ( 32000, 120, 8),
    "es":      ( 16000, 120, 8),
}


def render_slurm_resources(cfg: dict, stage: str) -> None:
    """Render mem / runtime / cpus number_inputs for a stage.

    Reads/writes cfg['slurm']['resources'][stage]. The webapp estimator's
    'Apply to session config' button populates the same block; values typed
    here win on save, so this is also the manual-override surface."""
    if stage not in _SLURM_DEFAULTS:
        return
    default_mem, default_runtime, default_cpus = _SLURM_DEFAULTS[stage]
    slurm = cfg.setdefault("slurm", {})
    resources = slurm.setdefault("resources", {})
    stage_res = resources.setdefault(stage, {})
    c1, c2, c3 = st.columns(3)
    stage_res["mem_mb"] = c1.number_input(
        "Memory (MB)",
        value=int(stage_res.get("mem_mb", default_mem)),
        min_value=1000,
        step=1000,
        key=f"{stage}_mem_mb_override",
        help="Per-job SLURM mem request. Estimator's 'Apply' button writes here; "
             "you can also override manually. Ensure the target partition can "
             "actually serve this size.",
    )
    stage_res["runtime"] = c2.number_input(
        "Runtime (min)",
        value=int(stage_res.get("runtime", default_runtime)),
        min_value=1,
        key=f"{stage}_runtime_override",
    )
    stage_res["cpus_per_task"] = c3.number_input(
        "CPUs per task",
        value=int(stage_res.get("cpus_per_task", default_cpus)),
        min_value=1,
        key=f"{stage}_cpus_override",
    )


def render_binning_controls(cfg: dict, stage: str) -> None:
    """Per-stage 'Bin-aware chunking' toggle + mode + preview table.

    The bins recipe (chunk_size/mem/runtime per bin) is populated by the
    estimator's 'Apply to session config' button. The UI toggles enabled +
    mode + num_bins; per-bin numbers are read-only here (edit config.yaml
    for fine-grained overrides).
    """
    if stage not in {"msa", "boltz", "esm", "esmfold"}:
        return
    stage_cfg = cfg.setdefault(stage, {})
    binning = stage_cfg.setdefault("binning", {})

    enabled = st.toggle(
        "Bin-aware chunking",
        value=bool(binning.get("enabled", False)),
        help="Partition sequences into length bins; each bin produces chunks "
             "with bin-specific SLURM mem and runtime. Recipe populated by "
             "the estimator (click 'Apply to session config' after enabling).",
        key=f"{stage}_binning_enabled",
    )
    binning["enabled"] = enabled
    if not enabled:
        return

    c1, c2, c3 = st.columns([1, 1, 1])
    mode = c1.selectbox(
        "Bin mode",
        options=["quantile", "thresholds"],
        index=0 if binning.get("mode", "quantile") == "quantile" else 1,
        help="quantile: cuts derived from your input length distribution. "
             "thresholds: explicit cuts (set in 'Length cuts' below).",
        key=f"{stage}_binning_mode",
    )
    binning["mode"] = mode
    if mode == "quantile":
        binning["num_bins"] = c2.number_input(
            "Number of bins",
            value=int(binning.get("num_bins", 6)),
            min_value=2,
            max_value=10,
            help="6 uses upper-tail-weighted cuts (q25/q50/q75/q90/q95). "
                 "Other values use evenly-spaced quantiles.",
            key=f"{stage}_binning_num_bins",
        )
    else:
        thresholds_str = ",".join(str(int(t)) for t in (binning.get("thresholds") or []))
        new_str = c2.text_input(
            "Length cuts",
            value=thresholds_str,
            help="Comma-separated. Example: 400,800,1200,1800 -> 5 bins.",
            key=f"{stage}_binning_thresholds",
        )
        try:
            binning["thresholds"] = [int(x.strip()) for x in new_str.split(",") if x.strip()]
            binning["num_bins"] = len(binning["thresholds"]) + 1
        except ValueError:
            st.warning("Thresholds must be integers separated by commas.")

    binning["chunks_per_bin"] = c3.number_input(
        "Chunks per bin",
        value=int(binning.get("chunks_per_bin", 1)),
        min_value=1,
        help="How many parallel chunks each non-empty bin is split into. "
             "Total parallel jobs ≈ num_bins × chunks_per_bin (capped at "
             "bin_count for sparse bins).",
        key=f"{stage}_binning_chunks_per_bin",
    )

    # Preview from the latest estimate (if any). last_estimate stores dict
    # form (asdict of StageEstimate), so bin_plan here is a dict or None.
    est = st.session_state.get("last_estimate", {}).get(stage)
    bin_plan = est.get("bin_plan") if isinstance(est, dict) else None
    if bin_plan and bin_plan.get("bins"):
        st.caption("Estimated plan (re-run estimator after changing input data):")
        rows = []
        total_chunks = 0
        total_runtime = 0
        cpb = bin_plan.get("chunks_per_bin", 1)
        for b in bin_plan["bins"]:
            n = b["num_seqs"]
            n_chunks = min(cpb, n) if n else 0
            cs = b.get("chunk_size") or (((n + n_chunks - 1) // n_chunks) if n_chunks else 0)
            total_chunks += n_chunks
            total_runtime += n_chunks * b["runtime_min"]
            rows.append({
                "bin": b["bin_idx"],
                "n": n,
                "L range": f"{b['len_lo']}-{b['len_hi']}",
                "chunks": n_chunks,
                "seqs/chunk": cs,
                "mem_mb": b["mem_mb"],
                "runtime_min": b["runtime_min"],
            })
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            f"Total chunks: {total_chunks} (≈ {bin_plan.get('num_bins', 0)} bins × "
            f"{cpb} chunks/bin), total runtime budget: {total_runtime} min "
            f"(thresholds: {bin_plan.get('thresholds', [])})"
        )
    else:
        st.caption("No bin plan yet — run the estimator to populate per-bin recipes.")


def render_estimate_panel(scan_result: dict, cfg: dict, session: Session,
                          key_prefix: str = "est") -> None:
    """Render the resource-estimate expander given a scan_directory() result.

    Computes input stats, calls the estimator for each enabled pipeline stage,
    shows a per-stage table, and offers an "Apply to session config" button.
    Caches stats + estimate in st.session_state for the Configuration tab.
    """
    file_type = scan_result.get("file_type")
    if file_type == "fasta":
        stats = compute_input_stats(fasta_results=scan_result.get("fasta_results", []))
    elif file_type == "yaml":
        stats = compute_input_stats(yaml_results=scan_result.get("yaml_results", []))
    else:
        return

    if stats.count == 0:
        return

    # GPU preferences from session_state (set by Configuration tab dropdowns).
    gpu_prefs = st.session_state.get("gpu_preferences", {})

    try:
        scaling = load_scaling_models()
        estimates = estimate_all_stages(stats, cfg, scaling, gpu_prefs)
    except Exception as exc:
        st.error(f"Resource estimate failed: {exc}")
        return

    # Cache for Configuration tab
    st.session_state["last_input_stats"] = stats.as_dict()
    st.session_state["last_estimate"] = {s: e.as_dict() for s, e in estimates.items()}

    if not estimates:
        st.info(
            f"Found {stats.count} valid {stats.file_type.upper()} file(s), "
            "but no pipeline stages are enabled. Toggle MSA / Boltz / ESM / "
            "ESMFold / ES above to see resource estimates."
        )
        return

    total_node_h = sum(e.total_node_hours for e in estimates.values())
    total_jobs = sum(e.num_chunks for e in estimates.values())

    with st.expander(
        f"Resource estimate — {total_node_h:.1f} node-hours across {total_jobs} jobs",
        expanded=True,
    ):
        st.caption(
            f"Based on {stats.count} sequence(s): "
            f"mean length {stats.mean_len:.0f}, p95 {stats.p95_len}, max {stats.max_len}."
        )

        rows = []
        any_notes = False
        for stage, e in estimates.items():
            rows.append({
                "Stage": stage.upper(),
                "Mem (GB)": round(e.mem_mb / 1024, 1),
                "Runtime (min)": e.runtime_min,
                "CPUs": e.cpus,
                "GPU": e.gpu_type or "-",
                "Partition": e.partition or "-",
                "Chunk size": e.chunk_size,
                "# Jobs": e.num_chunks,
                "Node-hours": e.total_node_hours,
            })
            if e.notes:
                any_notes = True
        st.dataframe(rows, width="stretch", hide_index=True)

        if any_notes:
            with st.expander("Notes from estimator", expanded=False):
                for stage, e in estimates.items():
                    for n in e.notes:
                        st.write(f"- **{stage}**: {n}")

        col_apply, col_info = st.columns([2, 3])
        with col_apply:
            if st.button(
                "Apply estimates to session config",
                key=f"{key_prefix}_apply",
                type="secondary",
                width="stretch",
            ):
                # Reuse estimator's writer; pass StageEstimate objects (we cached
                # dicts in session_state, so reload them as objects).
                try:
                    apply_estimate_to_config(session.config_path, estimates, backup=True)
                    touch_session(session.id)
                    st.success(
                        "Wrote slurm.resources.<stage> + chunk sizes to "
                        f"`{session.config_path.name}` (backup saved)."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not apply estimates: {exc}")
        with col_info:
            st.caption(
                "Writes per-stage mem/runtime/cpus/gpus to slurm.resources, "
                "partition under slurm.<stage>.partition, and chunk size into "
                "the stage's own block."
            )


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
        done = count_files(seq_dir, "*/msa/*.a3m")
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

    if pipeline.get("esm"):
        done = count_files(seq_dir, "*/esm/logits.npy")
        progress["ESM"] = (done, total)

    if pipeline.get("esmfold"):
        done = count_files(seq_dir, "*/esmfold/structure.pdb")
        progress["ESMFold"] = (done, total)

    if pipeline.get("es"):
        es_dir = output_dir / "es"
        done = count_files(es_dir, "*.csv")
        progress["ES"] = (done, total)

    return progress


def fastest_refresh(progress: dict) -> int | None:
    intervals = []
    for stage, (done, total) in progress.items():
        if total > 0 and done < total:
            intervals.append(REFRESH_INTERVALS.get(stage, 60))
    return min(intervals) if intervals else None


# ---------------------------------------------------------------------------
# Bootstrap: migrate legacy config and ensure at least one session exists
# ---------------------------------------------------------------------------
migrate_legacy()

registry = load_registry()
if not registry["sessions"]:
    create_session("Default")
    registry = load_registry()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ProtForge", layout="wide")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = get_active_session_id() or registry["sessions"][0]["id"]

# ---------------------------------------------------------------------------
# Sidebar — session management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Sessions")

    sessions = list_sessions()
    session_ids = [s["id"] for s in sessions]

    # Ensure active session is valid
    if st.session_state.active_session_id not in session_ids:
        st.session_state.active_session_id = session_ids[0] if session_ids else None

    # Session list
    for s_info in sessions:
        s_obj = get_session(s_info["id"])
        is_active = s_info["id"] == st.session_state.active_session_id
        running, _, _ = snakemake_status(s_obj)
        status_dot = "🟢" if running else "⚪"

        col_btn, col_status = st.columns([5, 1])
        with col_btn:
            label = f"**{s_info['name']}**" if is_active else s_info["name"]
            if st.button(
                label,
                key=f"switch_{s_info['id']}",
                width="stretch",
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_session_id = s_info["id"]
                set_active_session(s_info["id"])
                st.rerun()
        with col_status:
            st.markdown(status_dot)

    st.divider()

    # New session
    with st.expander("New Session", expanded=False):
        new_name = st.text_input("Session name", value="", key="new_session_name")
        clone_options = ["Empty config"] + [s["name"] for s in sessions]
        clone_choice = st.selectbox("Clone config from", options=clone_options, key="clone_source")

        if st.button("Create", key="create_session", width="stretch"):
            if not new_name.strip():
                st.error("Enter a session name.")
            else:
                base_config = None
                if clone_choice != "Empty config":
                    # Find the session to clone from
                    for s_info in sessions:
                        if s_info["name"] == clone_choice:
                            base_config = load_config(get_session(s_info["id"]))
                            break
                new_session = create_session(new_name.strip(), base_config=base_config)
                st.session_state.active_session_id = new_session.id
                set_active_session(new_session.id)
                st.rerun()

    # Session actions for active session
    if st.session_state.active_session_id:
        active_info = next((s for s in sessions if s["id"] == st.session_state.active_session_id), None)
        if active_info:
            with st.expander("Session Settings", expanded=False):
                new_label = st.text_input("Rename", value=active_info["name"], key="rename_input")
                if new_label != active_info["name"] and st.button("Rename", key="rename_btn"):
                    rename_session(active_info["id"], new_label)
                    st.rerun()

                st.caption(f"Created: {active_info.get('created', 'unknown')[:16]}")

                # Delete (only if not running and more than one session)
                s_obj = get_session(active_info["id"])
                running, _, _ = snakemake_status(s_obj)
                if len(sessions) > 1 and not running:
                    if st.button("Delete this session", type="secondary", key="delete_btn"):
                        st.session_state.confirm_delete = True

                    if st.session_state.get("confirm_delete"):
                        st.warning(f"Delete **{active_info['name']}** and all its config/logs?")
                        c1, c2 = st.columns(2)
                        if c1.button("Yes, delete", key="confirm_yes"):
                            delete_session(active_info["id"])
                            st.session_state.pop("confirm_delete", None)
                            new_registry = load_registry()
                            if new_registry["sessions"]:
                                st.session_state.active_session_id = new_registry["sessions"][0]["id"]
                            st.rerun()
                        if c2.button("Cancel", key="confirm_no"):
                            st.session_state.pop("confirm_delete", None)
                            st.rerun()
                elif running:
                    st.caption("Cannot delete a running session.")

# ---------------------------------------------------------------------------
# Get active session for main content
# ---------------------------------------------------------------------------
session = get_session(st.session_state.active_session_id)

# Header
col_title, col_info = st.columns([3, 1])
with col_title:
    active_name = next((s["name"] for s in sessions if s["id"] == session.id), session.id)
    st.title(f"ProtForge — {active_name}")
with col_info:
    st.caption(f"{USER}@{HOST}")

tab_config, tab_run, tab_monitor = st.tabs(["Configuration", "Run Pipeline", "Job Monitor"])

# ---------------------------------------------------------------------------
# Import dialog — opens as a popup from the Configuration tab
# ---------------------------------------------------------------------------
@st.dialog("Import Data", width="large")
def import_dialog():
    """Import UI: browse & copy from cluster, generate from CSV/TSV, or random mutations.

    Files are written into the FASTA/YAML directory from the session config.
    If that path is empty or doesn't exist, it is created.
    """
    # Resolve destination from the current session config
    cfg = load_config(session)
    inp = cfg.get("input", {})
    fasta_dir = inp.get("fasta_dir", "")
    yaml_dir = inp.get("yaml_dir", "")

    import_mode = st.radio(
        "Import mode",
        ["Browse directory", "Generate from CSV/TSV", "Random mutations"],
        horizontal=True,
        key="dlg_import_mode",
    )

    def _get_dest(file_type: str) -> Path | None:
        """Get the destination directory from config, or ask user to set it."""
        dest_str = fasta_dir if file_type == "fasta" else yaml_dir
        if not dest_str:
            st.error(
                f"No {'FASTA' if file_type == 'fasta' else 'YAML'} directory set in config. "
                "Set it in the Input / Output section first, then re-open Import."
            )
            return None
        dest = Path(dest_str)
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    def _update_config_after_import(file_type: str):
        """Update pipeline.msa and clear the other input path."""
        cfg_now = load_config(session)
        if "pipeline" not in cfg_now:
            cfg_now["pipeline"] = {}
        if file_type == "fasta":
            cfg_now["pipeline"]["msa"] = True
            cfg_now.get("input", {}).pop("yaml_dir", None)
        else:
            cfg_now["pipeline"]["msa"] = False
            cfg_now.get("input", {}).pop("fasta_dir", None)
        save_config(session, cfg_now)

    # =============================================================
    # MODE 1: Browse directory — copy valid files to input path
    # =============================================================
    if import_mode == "Browse directory":
        st.caption("Browse the cluster, validate files, and copy valid ones into the configured input directory.")

        if "browse_path" not in st.session_state:
            st.session_state.browse_path = os.environ.get("HOME", "/")

        col_path, col_go = st.columns([5, 1])
        with col_path:
            typed_path = st.text_input(
                "Source directory", value=st.session_state.browse_path, key="dlg_dir_path",
            )
        with col_go:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Go", width="stretch", key="dlg_go"):
                if Path(typed_path).is_dir():
                    st.session_state.browse_path = typed_path
                    st.rerun()
                else:
                    st.error("Not a valid directory.")

        browse_dir = Path(st.session_state.browse_path)

        # Breadcrumb
        parts = browse_dir.parts
        breadcrumb_cols = st.columns(min(len(parts), 10))
        for i, part in enumerate(parts[: len(breadcrumb_cols)]):
            with breadcrumb_cols[i]:
                label = part if part != "/" else "/"
                if st.button(label, key=f"dlg_bc_{i}", width="stretch"):
                    target = Path(*parts[: i + 1]) if i > 0 else Path("/")
                    st.session_state.browse_path = str(target)
                    st.rerun()

        # Subdirectories
        if browse_dir.is_dir():
            try:
                subdirs = sorted([d for d in browse_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
            except PermissionError:
                subdirs = []
                st.error("Permission denied.")

            if subdirs:
                st.markdown("**Subdirectories:**")
                for row_start in range(0, len(subdirs), 4):
                    row_dirs = subdirs[row_start : row_start + 4]
                    cols = st.columns(4)
                    for j, d in enumerate(row_dirs):
                        with cols[j]:
                            if st.button(f"📁 {d.name}", key=f"dlg_sd_{d.name}", width="stretch"):
                                st.session_state.browse_path = str(d)
                                st.rerun()

            st.divider()
            st.markdown(f"**Source:** `{browse_dir}`")

            if st.button("Scan & Validate", type="primary", width="stretch", key="dlg_scan"):
                with st.spinner("Scanning files..."):
                    st.session_state.scan_result = scan_directory(browse_dir)
                    st.session_state.scan_path = str(browse_dir)

            if "scan_result" in st.session_state and st.session_state.get("scan_path") == str(browse_dir):
                result = st.session_state.scan_result

                if result["file_type"] == "none":
                    st.warning("No FASTA (.fasta, .fa) or YAML (.yaml, .yml) files found.")
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total", result["total_files"])
                    c2.metric("Valid", result["valid_count"])
                    c3.metric("Invalid", result["invalid_count"])
                    c4.metric("Type", result["file_type"].upper())

                    if result["file_type"] == "mixed":
                        st.warning("Mixed FASTA and YAML — choose one type per directory.")

                    if result["fasta_results"]:
                        with st.expander(f"FASTA files ({len(result['fasta_results'])})", expanded=result["invalid_count"] > 0):
                            rows = [{
                                "File": r["filename"],
                                "Status": "Valid" if r["valid"] else "INVALID",
                                "Sequences": r["num_sequences"],
                                "Residues": r["total_residues"],
                                "Errors": "; ".join(r["errors"]) if r["errors"] else "",
                            } for r in result["fasta_results"]]
                            st.dataframe(rows, width="stretch", hide_index=True)

                    if result["yaml_results"]:
                        with st.expander(f"YAML files ({len(result['yaml_results'])})", expanded=result["invalid_count"] > 0):
                            rows = [{
                                "File": r["filename"],
                                "Status": "Valid" if r["valid"] else "INVALID",
                                "Seq ID": r["sequence_id"],
                                "Length": r["sequence_length"],
                                "Has MSA": "Yes" if r["has_msa"] else "No",
                                "Errors": "; ".join(r["errors"]) if r["errors"] else "",
                            } for r in result["yaml_results"]]
                            st.dataframe(rows, width="stretch", hide_index=True)

                    if result["valid_count"] > 0 and result["file_type"] in ("fasta", "yaml"):
                        st.divider()
                        render_estimate_panel(result, cfg, session, key_prefix="dlg_browse")

                    st.divider()
                    if result["valid_count"] > 0 and result["file_type"] in ("fasta", "yaml"):
                        # Show destination
                        dest_path = fasta_dir if result["file_type"] == "fasta" else yaml_dir
                        if dest_path:
                            st.info(f"**Destination:** `{dest_path}`")
                        if result["invalid_count"] > 0:
                            st.warning(f"{result['invalid_count']} invalid file(s) — only valid files will be copied.")

                        apply_label = f"Copy {result['valid_count']} valid files to input directory"
                        if st.button(apply_label, type="primary", width="stretch", key="dlg_apply_dir"):
                            dest = _get_dest(result["file_type"])
                            if dest is not None:
                                copied = copy_valid_files(result, dest)
                                _update_config_after_import(result["file_type"])
                                st.success(f"Copied {copied} files to `{dest}`")
                                st.rerun()

                    elif result["file_type"] == "mixed":
                        st.info("Separate FASTA and YAML files into different directories.")
        else:
            st.error(f"Directory not found: `{browse_dir}`")

    # =============================================================
    # MODE 2: Generate from CSV/TSV
    # =============================================================
    elif import_mode == "Generate from CSV/TSV":
        st.caption(
            "Upload a CSV/TSV with either:\n"
            "- **Mutations**: `aaMutations` column + reference sequence (`SA108D:SN144D`)\n"
            "- **Sequences**: `name` and `sequence` columns"
        )

        uploaded_csv = st.file_uploader("Upload CSV or TSV", type=["csv", "tsv", "txt"], key="dlg_csv")

        if uploaded_csv is not None:
            import pandas as pd

            sep = "," if uploaded_csv.name.lower().endswith(".csv") else "\t"
            try:
                df = pd.read_csv(uploaded_csv, sep=sep)
            except Exception as e:
                st.error(f"Failed to parse: {e}")
                df = None

            if df is not None:
                st.markdown(f"**Loaded:** {len(df)} rows, columns: `{', '.join(df.columns)}`")
                with st.expander("Preview", expanded=True):
                    st.dataframe(df.head(20), width="stretch", hide_index=True)

                has_mutations = "aaMutations" in df.columns
                has_sequences = "name" in df.columns and "sequence" in df.columns

                if has_mutations and has_sequences:
                    csv_mode = st.radio("Choose mode:", ["mutations", "sequences"], key="dlg_csv_mode")
                elif has_mutations:
                    csv_mode = "mutations"
                    st.info("Detected **mutations mode**.")
                elif has_sequences:
                    csv_mode = "sequences"
                    st.info("Detected **sequences mode**.")
                else:
                    st.error("CSV needs `aaMutations` or `name` + `sequence` columns.")
                    csv_mode = None

                if csv_mode:
                    file_type = st.selectbox("Output format", ["fasta", "yaml"], key="dlg_csv_ft",
                                             help="fasta: MSA pipeline; yaml: skip MSA")

                    ref_sequence = None
                    if csv_mode == "mutations":
                        st.markdown("**Reference sequence** (required)")
                        ref_method = st.radio("Provide via:", ["Paste sequence", "Upload FASTA"], horizontal=True, key="dlg_ref_m")
                        if ref_method == "Paste sequence":
                            ref_sequence = st.text_area("Reference sequence", height=80, key="dlg_ref_seq").strip().replace("\n", "").replace(" ", "")
                        else:
                            ref_upload = st.file_uploader("Reference FASTA", type=["fasta", "fa"], key="dlg_ref_fa")
                            if ref_upload is not None:
                                ref_text = ref_upload.read().decode("utf-8")
                                ref_lines = [l.strip() for l in ref_text.splitlines() if l.strip() and not l.startswith(">")]
                                ref_sequence = "".join(ref_lines)

                        if ref_sequence:
                            from validate import VALID_AAS as _VA
                            inv = set(ref_sequence.upper()) - _VA
                            if inv:
                                st.error(f"Invalid characters: {', '.join(sorted(inv))}")
                                ref_sequence = None
                            else:
                                st.success(f"Reference: {len(ref_sequence)} residues")

                    # Show destination
                    dest_str = fasta_dir if file_type == "fasta" else yaml_dir
                    if dest_str:
                        st.info(f"**Destination:** `{dest_str}`")

                    can_gen = csv_mode == "sequences" or (csv_mode == "mutations" and ref_sequence)
                    if st.button("Generate files", type="primary", disabled=not can_gen, width="stretch", key="dlg_gen_csv"):
                        output_dir = _get_dest(file_type)
                        if output_dir is not None:
                            from validate import VALID_AAS as _VA
                            errors = []
                            generated = 0

                            if csv_mode == "sequences":
                                for _, row in df.iterrows():
                                    name = str(row["name"]).strip()
                                    seq = str(row["sequence"]).strip()
                                    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
                                    if not safe_name:
                                        continue
                                    if set(seq.upper()) - _VA:
                                        errors.append(f"{name}: invalid chars")
                                        continue
                                    if file_type == "fasta":
                                        (output_dir / f"{safe_name}.fasta").write_text(f">{safe_name}\n{seq}\n")
                                    else:
                                        (output_dir / f"{safe_name}.yaml").write_text(yaml.dump({
                                            "version": 1,
                                            "sequences": [{"protein": {"id": safe_name, "sequence": seq, "msa": "empty"}}],
                                        }, default_flow_style=False, sort_keys=False))
                                    generated += 1
                            else:
                                import re
                                for _, row in df.iterrows():
                                    mut_str = str(row["aaMutations"]).strip()
                                    if pd.isna(row["aaMutations"]) or not mut_str:
                                        continue
                                    mutated = ref_sequence
                                    valid = True
                                    try:
                                        for m in mut_str.split(":"):
                                            match = re.match(r"^S([A-Za-z])(\d+)([A-Za-z])$", m)
                                            if not match:
                                                errors.append(f"Invalid format: {m}")
                                                valid = False
                                                break
                                            src, pos_str, dest = match.groups()
                                            pos = int(pos_str)
                                            if pos < 1 or pos > len(ref_sequence):
                                                errors.append(f"{m}: position out of range")
                                                valid = False
                                                break
                                            idx = pos - 1
                                            if mutated[idx].upper() != src.upper():
                                                errors.append(f"{m}: expected {src} at pos {pos}, found {mutated[idx]}")
                                                valid = False
                                                break
                                            mutated = mutated[:idx] + dest + mutated[idx + 1:]
                                    except Exception as e:
                                        errors.append(f"{mut_str}: {e}")
                                        valid = False
                                    if not valid:
                                        continue
                                    safe_name = mut_str.replace(":", "_")
                                    if file_type == "fasta":
                                        (output_dir / f"{safe_name}.fasta").write_text(f">{safe_name}\n{mutated}\n")
                                    else:
                                        (output_dir / f"{safe_name}.yaml").write_text(yaml.dump({
                                            "version": 1,
                                            "sequences": [{"protein": {"id": safe_name, "sequence": mutated, "msa": "empty"}}],
                                        }, default_flow_style=False, sort_keys=False))
                                    generated += 1

                            if generated > 0:
                                _update_config_after_import(file_type)
                                st.success(f"Generated {generated} files in `{output_dir}`")
                                render_estimate_panel(
                                    scan_directory(output_dir),
                                    load_config(session),
                                    session,
                                    key_prefix="dlg_csv",
                                )
                            else:
                                st.error("No files generated.")
                            if errors:
                                with st.expander(f"Errors ({len(errors)})"):
                                    for e in errors[:50]:
                                        st.text(e)

    # =============================================================
    # MODE 3: Random mutations
    # =============================================================
    elif import_mode == "Random mutations":
        st.caption("Provide a reference sequence and generate random single or multi-site mutants.")

        ref_seq_rand = st.text_area(
            "Reference amino acid sequence", height=100,
            help="Paste the wildtype protein sequence (amino acids only)",
            key="dlg_rand_ref",
        ).strip().replace("\n", "").replace(" ", "")

        if ref_seq_rand:
            from validate import VALID_AAS as _VA
            inv = set(ref_seq_rand.upper()) - _VA
            if inv:
                st.error(f"Invalid characters: {', '.join(sorted(inv))}")
            else:
                st.success(f"Reference: {len(ref_seq_rand)} residues")

                c1, c2, c3 = st.columns(3)
                n_mutants = c1.number_input("Number of mutants", value=100, min_value=1, max_value=100000, key="dlg_n_mut")
                n_mutations = c2.number_input("Mutations per sequence", value=1, min_value=1, max_value=20, key="dlg_n_muts")
                seed = c3.number_input("Random seed", value=42, min_value=0, key="dlg_seed")

                file_type_rand = st.selectbox("Output format", ["fasta", "yaml"], key="dlg_rand_ft",
                                              help="fasta: MSA pipeline; yaml: skip MSA")

                # Show destination
                dest_str = fasta_dir if file_type_rand == "fasta" else yaml_dir
                if dest_str:
                    st.info(f"**Destination:** `{dest_str}`")

                if st.button("Generate random mutants", type="primary", width="stretch", key="dlg_gen_rand"):
                    output_dir = _get_dest(file_type_rand)
                    if output_dir is not None:
                        import random as _random
                        _random.seed(seed)

                        aa_list = list(_VA)
                        seq_len = len(ref_seq_rand)
                        generated = 0
                        seen = set()
                        with st.spinner(f"Generating {n_mutants} mutants..."):
                            attempts = 0
                            max_attempts = n_mutants * 10
                            while generated < n_mutants and attempts < max_attempts:
                                attempts += 1
                                positions = _random.sample(range(seq_len), min(n_mutations, seq_len))
                                mutated = list(ref_seq_rand)
                                mut_parts = []
                                for pos in sorted(positions):
                                    src = ref_seq_rand[pos]
                                    candidates = [aa for aa in aa_list if aa != src]
                                    dest = _random.choice(candidates)
                                    mutated[pos] = dest
                                    mut_parts.append(f"S{src}{pos + 1}{dest}")
                                mut_key = ":".join(mut_parts)
                                if mut_key in seen:
                                    continue
                                seen.add(mut_key)
                                mutated_seq = "".join(mutated)
                                safe_name = mut_key.replace(":", "_")
                                if file_type_rand == "fasta":
                                    (output_dir / f"{safe_name}.fasta").write_text(f">{safe_name}\n{mutated_seq}\n")
                                else:
                                    (output_dir / f"{safe_name}.yaml").write_text(yaml.dump({
                                        "version": 1,
                                        "sequences": [{"protein": {"id": safe_name, "sequence": mutated_seq, "msa": "empty"}}],
                                    }, default_flow_style=False, sort_keys=False))
                                generated += 1

                        if generated < n_mutants:
                            st.warning(f"Only {generated}/{n_mutants} unique mutants (sequence space too small).")

                        _update_config_after_import(file_type_rand)
                        st.success(f"Generated {generated} mutants in `{output_dir}`")
                        render_estimate_panel(
                            scan_directory(output_dir),
                            load_config(session),
                            session,
                            key_prefix="dlg_rand",
                        )


# ---------------------------------------------------------------------------
# Advanced Boltz options dialog — extra `boltz predict` flags, all opt-in
# ---------------------------------------------------------------------------
@st.dialog("Advanced Boltz Options", width="large")
def boltz_advanced_dialog():
    """All optional `boltz predict` CLI flags. Only flags marked Enable are passed."""
    cfg = load_config(session)
    boltz = cfg.get("boltz", {})
    adv = boltz.get("advanced", {}) or {}

    st.caption(
        "Optional flags for `boltz predict`. Only options marked **Enable** are passed; "
        "the rest fall back to Boltz defaults. "
        "See [Boltz prediction docs](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md)."
    )

    new_adv: dict = {}
    flag_options = {
        "affinity_mw_correction", "subsample_msa", "no_kernels",
        "use_potentials", "write_full_pae", "write_full_pde",
    }

    def _typed_opt(key: str, label: str, default, help_text: str,
                   kind: str = "int", choices: list[str] | None = None):
        cur = adv.get(key)
        c1, c2 = st.columns([1, 4])
        enabled = c1.checkbox("Enable", value=cur is not None, key=f"adv_en_{key}")
        with c2:
            if kind == "int":
                v = st.number_input(
                    label, value=int(cur) if cur is not None else int(default), step=1,
                    disabled=not enabled, help=help_text, key=f"adv_v_{key}",
                )
                if enabled:
                    new_adv[key] = int(v)
            elif kind == "float":
                v = st.number_input(
                    label, value=float(cur) if cur is not None else float(default),
                    step=0.001, format="%.3f",
                    disabled=not enabled, help=help_text, key=f"adv_v_{key}",
                )
                if enabled:
                    new_adv[key] = float(v)
            elif kind == "select":
                opts = choices or []
                idx = opts.index(cur) if cur in opts else opts.index(default)
                v = st.selectbox(
                    label, options=opts, index=idx,
                    disabled=not enabled, help=help_text, key=f"adv_v_{key}",
                )
                if enabled:
                    new_adv[key] = v
            elif kind == "str":
                v = st.text_input(
                    label, value=str(cur) if cur is not None else str(default),
                    disabled=not enabled, help=help_text, key=f"adv_v_{key}",
                )
                if enabled and str(v).strip():
                    new_adv[key] = str(v).strip()

    def _flag_opt(key: str, label: str, help_text: str):
        v = st.checkbox(label, value=bool(adv.get(key, False)),
                        help=help_text, key=f"adv_v_{key}")
        if v:
            new_adv[key] = True

    st.markdown("##### Sampling")
    _typed_opt("sampling_steps", "sampling_steps", 200, "Number of sampling steps", "int")
    _typed_opt("step_scale", "step_scale", 1.638,
               "Diffusion temperature; range [1-2] for diversity control", "float")
    _typed_opt("max_parallel_samples", "max_parallel_samples", 5,
               "Maximum samples to predict in parallel", "int")
    _typed_opt("output_format", "output_format", "mmcif",
               "Output structure format", "select", ["mmcif", "pdb"])
    _typed_opt("num_workers", "num_workers", 2, "Dataloader worker threads", "int")
    _typed_opt("method", "method", "", "Prediction method selection", "str")
    _typed_opt("preprocessing_threads", "preprocessing-threads", 4,
               "Preprocessing thread count (default: CPU count)", "int")

    st.markdown("##### Affinity")
    _flag_opt("affinity_mw_correction", "affinity_mw_correction",
              "Apply molecular weight correction to affinity")
    _typed_opt("sampling_steps_affinity", "sampling_steps_affinity", 200,
               "Affinity prediction sampling steps", "int")
    _typed_opt("diffusion_samples_affinity", "diffusion_samples_affinity", 5,
               "Affinity diffusion samples", "int")
    _typed_opt("affinity_checkpoint", "affinity_checkpoint", "",
               "Custom affinity model checkpoint path", "str")

    st.markdown("##### MSA")
    _typed_opt("max_msa_seqs", "max_msa_seqs", 8192, "Maximum MSA sequences", "int")
    _flag_opt("subsample_msa", "subsample_msa", "Enable MSA subsampling")
    _typed_opt("num_subsampled_msa", "num_subsampled_msa", 1024,
               "Number of sequences after subsampling", "int")
    _typed_opt("msa_pairing_strategy", "msa_pairing_strategy", "greedy",
               "MSA pairing strategy", "select", ["greedy", "complete"])

    st.markdown("##### Other")
    _flag_opt("no_kernels", "no_kernels",
              "Disable trifast kernels for triangular updates")
    _flag_opt("use_potentials", "use_potentials",
              "Enable inference-time potentials")
    _flag_opt("write_full_pae", "write_full_pae", "Save full PAE matrix file")
    _flag_opt("write_full_pde", "write_full_pde", "Save full PDE matrix file")
    _typed_opt("checkpoint", "checkpoint", "",
               "Custom model checkpoint path (overrides default Boltz-2)", "str")

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("Save", type="primary", width="stretch", key="adv_save"):
        if new_adv:
            boltz["advanced"] = new_adv
        else:
            boltz.pop("advanced", None)
        cfg["boltz"] = boltz
        save_config(session, cfg)
        st.success(f"Saved {len(new_adv)} advanced option(s).")
        st.rerun()
    if c2.button("Reset all", width="stretch", key="adv_reset"):
        boltz.pop("advanced", None)
        cfg["boltz"] = boltz
        save_config(session, cfg)
        st.success("Reset advanced options.")
        st.rerun()


# =========================================================================
# TAB 1: Configuration
# =========================================================================
with tab_config:
    cfg = load_config(session)

    with st.expander("Pipeline Stages", expanded=True):
        pipeline = cfg.get("pipeline", {})
        cols = st.columns(5)
        pipeline["msa"] = cols[0].toggle("MSA", value=pipeline.get("msa", True))
        pipeline["boltz"] = cols[1].toggle("Boltz", value=pipeline.get("boltz", True))
        pipeline["esm"] = cols[2].toggle("ESM", value=pipeline.get("esm", True))
        pipeline["esmfold"] = cols[3].toggle("ESMFold", value=pipeline.get("esmfold", False))
        pipeline["es"] = cols[4].toggle("ES", value=pipeline.get("es", False))
        cfg["pipeline"] = pipeline

    with st.expander("Input / Output", expanded=True):
        inp = cfg.get("input", {})
        out = cfg.get("output", {})

        col_fasta, col_import = st.columns([5, 1])
        with col_fasta:
            inp["fasta_dir"] = st.text_input("FASTA directory", value=inp.get("fasta_dir", ""))
        with col_import:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Import", width="stretch", key="open_import"):
                # Save current input values so the dialog picks them up
                cfg["input"] = inp
                save_config(session, cfg)
                import_dialog()

        yaml_dir = inp.get("yaml_dir", "")
        inp["yaml_dir"] = st.text_input(
            "YAML directory (when MSA is off)", value=yaml_dir,
            help="Set this when pipeline.msa is false",
        )
        if not inp["yaml_dir"]:
            inp.pop("yaml_dir", None)
        out["parent_dir"] = st.text_input("Output directory", value=out.get("parent_dir", ""))
        cfg["input"] = inp
        cfg["output"] = out

        # Auto-scan + inline resource estimate.
        # Picks fasta_dir if MSA is on (or no preference set), otherwise yaml_dir.
        scan_path = ""
        if cfg.get("pipeline", {}).get("msa", True):
            scan_path = inp.get("fasta_dir", "") or inp.get("yaml_dir", "")
        else:
            scan_path = inp.get("yaml_dir", "") or inp.get("fasta_dir", "")

        if scan_path:
            res = autoscan_directory(scan_path)
            if res is None:
                st.caption(f"`{scan_path}` is not a directory (yet)")
            elif res["file_type"] == "none":
                st.caption(f"No FASTA/YAML files found in `{scan_path}`")
            elif res["file_type"] == "mixed":
                st.warning(
                    f"Both FASTA and YAML files in `{scan_path}` — keep only one type."
                )
            elif res["valid_count"] == 0:
                st.warning(
                    f"Found {res['total_files']} file(s) in `{scan_path}` but none "
                    "passed validation. Open Import for per-file errors."
                )
            else:
                # Per-file counts + sequence length distribution
                if res["file_type"] == "fasta":
                    pre_stats = compute_input_stats(fasta_results=res["fasta_results"])
                else:
                    pre_stats = compute_input_stats(yaml_results=res["yaml_results"])

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Valid sequences", pre_stats.count)
                m2.metric("Invalid files", res["invalid_count"])
                m3.metric("Total residues", f"{pre_stats.total_residues:,}")
                m4.metric("Type", res["file_type"].upper())

                if pre_stats.count > 0:
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Min length", pre_stats.min_len)
                    s2.metric("Mean length", f"{pre_stats.mean_len:.0f}")
                    s3.metric("p95 length", pre_stats.p95_len)
                    s4.metric("Max length", pre_stats.max_len)

                if st.button("Re-scan", key="config_rescan"):
                    _cached_scan.clear()
                    st.rerun()
                render_estimate_panel(res, cfg, session, key_prefix="config_inline")

    with st.expander("MSA Settings"):
        msa = cfg.get("msa", {})
        msa["max_files_per_job"] = st.number_input("Files per job", value=msa.get("max_files_per_job", 25), min_value=1)
        render_chunk_recommendation("msa")
        render_gpu_preference("msa")
        render_slurm_resources(cfg, "msa")
        render_binning_controls(cfg, "msa")
        msa["mmseq2_db"] = st.text_input("MMseqs2 DB", value=msa.get("mmseq2_db", ""))
        msa["colabfold_db"] = st.text_input("ColabFold DB", value=msa.get("colabfold_db", ""))
        msa["colabfold_bin"] = st.text_input("ColabFold bin", value=msa.get("colabfold_bin", ""))
        cfg["msa"] = msa

    with st.expander("Boltz Settings"):
        boltz = cfg.get("boltz", {})
        c1, c2 = st.columns(2)
        boltz["max_files_per_job"] = c1.number_input("Files per job ", value=boltz.get("max_files_per_job", 25), min_value=1)
        boltz["num_runs"] = c2.number_input("Runs per sequence", value=boltz.get("num_runs", 1), min_value=1)
        render_chunk_recommendation("boltz")
        render_gpu_preference("boltz")
        render_slurm_resources(cfg, "boltz")
        render_binning_controls(cfg, "boltz")
        c1, c2 = st.columns(2)
        boltz["recycling_steps"] = c1.number_input("Recycling steps", value=boltz.get("recycling_steps", 10), min_value=1)
        boltz["diffusion_samples"] = c2.number_input("Diffusion samples", value=boltz.get("diffusion_samples", 25), min_value=1)

        _ds_max = int(boltz["diffusion_samples"])
        _current_save = boltz.get("samples_to_save", 1)
        _save_all_default = _current_save == "all"
        c1, c2 = st.columns(2)
        _save_all = c1.toggle(
            "Save all diffusion samples",
            value=_save_all_default,
            help="Keep every generated sample. Otherwise keep the top-N best by confidence "
                 "(model_0..model_(N-1)).",
            key="boltz_save_all",
        )
        if _save_all:
            boltz["samples_to_save"] = "all"
            c2.number_input("Top N to save", value=_ds_max, min_value=1, max_value=_ds_max,
                            disabled=True, key="boltz_samples_to_save_disabled")
        else:
            _n_default = 1 if _current_save == "all" else int(_current_save)
            _n_default = min(max(_n_default, 1), _ds_max)
            boltz["samples_to_save"] = int(c2.number_input(
                "Top N to save",
                value=_n_default, min_value=1, max_value=_ds_max,
                help=f"Save top-N best models (max = diffusion samples = {_ds_max}).",
                key="boltz_samples_to_save",
            ))

        boltz["delete_msa_after_processing"] = st.toggle(
            "Delete MSA after Boltz", value=boltz.get("delete_msa_after_processing", False),
        )
        _boltz_use_cutoff = st.toggle(
            "Skip sequences over a length cutoff",
            value=boltz.get("max_seq_len") is not None,
            help="Sequences longer than the cutoff are dropped before chunking. "
                 "Skipped entries are listed in <output>/boltz_chunks/skipped_sequences.tsv. "
                 "Useful for avoiding the long-protein organize failure (~1800aa+).",
            key="boltz_use_max_seq_len",
        )
        if _boltz_use_cutoff:
            boltz["max_seq_len"] = st.number_input(
                "Max sequence length (residues)",
                value=int(boltz.get("max_seq_len") or 1500),
                min_value=1,
                key="boltz_max_seq_len",
            )
        else:
            boltz["max_seq_len"] = None
        boltz["cache_dir"] = st.text_input("Boltz cache dir", value=boltz.get("cache_dir", ""))
        boltz["env_path"] = st.text_input("Boltz env path", value=boltz.get("env_path", ""))

        adv_count = len(boltz.get("advanced", {}) or {})
        adv_label = (
            f"Advanced Boltz Options ({adv_count} set)"
            if adv_count else "Advanced Boltz Options..."
        )
        if st.button(adv_label, key="open_boltz_adv"):
            cfg["boltz"] = boltz
            save_config(session, cfg)
            boltz_advanced_dialog()

        cfg["boltz"] = boltz

    with st.expander("ESM Settings"):
        esm = cfg.get("esm", {})
        esm["num_chunks"] = st.number_input("Chunks", value=esm.get("num_chunks", 1), min_value=1)
        render_chunk_recommendation("esm")
        render_gpu_preference("esm")
        render_slurm_resources(cfg, "esm")
        render_binning_controls(cfg, "esm")
        esm["env_path"] = st.text_input("ESM env path", value=esm.get("env_path", ""))
        esm["cache_dir"] = st.text_input("ESM cache dir", value=esm.get("cache_dir", ""))
        cfg["esm"] = esm

    with st.expander("ESMFold Settings"):
        esmfold = cfg.get("esmfold", {})
        current_type = esmfold.get("input_type", "yaml")
        esmfold["input_type"] = st.selectbox(
            "Input source",
            options=["yaml", "fasta"],
            index=0 if current_type == "yaml" else 1,
            help="'yaml' reads from sequences/ (post-MSA) or input.yaml_dir. "
                 "'fasta' reads directly from input.fasta_dir (independent of MSA/Boltz).",
            key="esmfold_input_type",
        )
        esmfold["num_chunks"] = st.number_input(
            "Chunks ",
            value=esmfold.get("num_chunks", 1),
            min_value=1,
            key="esmfold_num_chunks",
        )
        render_chunk_recommendation("esmfold")
        render_gpu_preference("esmfold")
        render_slurm_resources(cfg, "esmfold")
        render_binning_controls(cfg, "esmfold")
        esmfold["array_max_concurrency"] = st.number_input(
            "Max concurrent array tasks",
            value=esmfold.get("array_max_concurrency", 10),
            min_value=1,
            key="esmfold_array_max_concurrency",
        )
        esmfold["env_path"] = st.text_input(
            "ESMFold env path",
            value=esmfold.get(
                "env_path",
                "/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge/envs/esmfold",
            ),
            key="esmfold_env_path",
            help="Conda env prefix to `conda activate`. Shared lab default: "
                 "/n/holylfs06/LABS/bsabatini_lab/Everyone/protforge/envs/esmfold",
        )
        esmfold["cache_dir"] = st.text_input(
            "ESMFold cache dir (HF_HOME root)",
            value=esmfold.get(
                "cache_dir",
                "/n/holylfs06/LABS/bsabatini_lab/Everyone/esm_models_cache",
            ),
            key="esmfold_cache_dir",
            help="Must contain hub/models--facebook--esmfold_v1. Pre-populate with "
                 "scripts/download_esmfold.py on a login node.",
        )

        _esmfold_use_cutoff = st.toggle(
            "Skip sequences over a length cutoff",
            value=esmfold.get("max_seq_len") is not None,
            help="Sequences longer than the cutoff are dropped before chunking. "
                 "Skipped entries are listed in <output>/esmfold_chunks/skipped_sequences.tsv.",
            key="esmfold_use_max_seq_len",
        )
        if _esmfold_use_cutoff:
            esmfold["max_seq_len"] = st.number_input(
                "Max sequence length (residues) ",
                value=int(esmfold.get("max_seq_len") or 2500),
                min_value=1,
                key="esmfold_max_seq_len",
            )
        else:
            esmfold["max_seq_len"] = None

        st.markdown("---")
        st.markdown("**Smart binning** — group long sequences into separate jobs")
        esmfold["bin_by_length"] = st.toggle(
            "Bin sequences by length",
            value=bool(esmfold.get("bin_by_length", False)),
            help="Partition inputs into a 'short' pool (L < threshold) and a 'long' pool "
                 "(L ≥ threshold) and chunk each pool independently. Long-pool jobs get "
                 "their own SLURM mem/time. Disable to use the legacy single-pool chunker.",
            key="esmfold_bin_by_length",
        )
        c1, c2, c3 = st.columns(3)
        esmfold["length_threshold"] = c1.number_input(
            "Length threshold (aa)",
            value=int(esmfold.get("length_threshold", 1200)),
            min_value=1,
            help="Sequences with L ≥ threshold go to the long pool. 1200 matches the "
                 "H100 80GB OOM ceiling observed in calibration.",
            key="esmfold_length_threshold",
        )
        esmfold["num_chunks_short"] = c2.number_input(
            "Chunks (short pool)",
            value=int(esmfold.get("num_chunks_short", esmfold.get("num_chunks", 1))),
            min_value=1,
            key="esmfold_num_chunks_short",
        )
        esmfold["num_chunks_long"] = c3.number_input(
            "Chunks (long pool)",
            value=int(esmfold.get("num_chunks_long", esmfold.get("num_chunks", 1))),
            min_value=1,
            key="esmfold_num_chunks_long",
        )
        c1, c2 = st.columns(2)
        esmfold["mem_short_mb"] = c1.number_input(
            "Short-pool mem (MB)",
            value=int(esmfold.get("mem_short_mb", 32000)),
            min_value=1000,
            step=1000,
            key="esmfold_mem_short_mb",
        )
        esmfold["mem_long_mb"] = c2.number_input(
            "Long-pool mem (MB)",
            value=int(esmfold.get("mem_long_mb", 80000)),
            min_value=1000,
            step=1000,
            help="Long-pool jobs run the trunk-chunked path; on H100 80GB give them headroom.",
            key="esmfold_mem_long_mb",
        )
        c1, c2 = st.columns(2)
        esmfold["time_short_min"] = c1.number_input(
            "Short-pool runtime (min)",
            value=int(esmfold.get("time_short_min", 60)),
            min_value=1,
            key="esmfold_time_short_min",
        )
        esmfold["time_long_min"] = c2.number_input(
            "Long-pool runtime (min)",
            value=int(esmfold.get("time_long_min", 240)),
            min_value=1,
            help="Trunk-chunked path is ~50% slower per residue, so long-pool jobs need more wall time.",
            key="esmfold_time_long_min",
        )

        st.markdown("---")
        st.markdown("**Trunk chunking** — per-sequence OOM guard inside `run_esmfold.py`")
        c1, c2 = st.columns(2)
        esmfold["chunk_size_threshold"] = c1.number_input(
            "Chunk-size threshold (aa)",
            value=int(esmfold.get("chunk_size_threshold", 1200)),
            min_value=1,
            help="When a sequence's L ≥ threshold, model.trunk.set_chunk_size(chunk_size) "
                 "is invoked at fold time to avoid OOM (slower but safe).",
            key="esmfold_chunk_size_threshold",
        )
        esmfold["chunk_size"] = c2.number_input(
            "Chunk size",
            value=int(esmfold.get("chunk_size", 64)),
            min_value=1,
            key="esmfold_chunk_size",
        )

        cfg["esmfold"] = esmfold

    with st.expander("ES Settings"):
        es = cfg.get("es", {})
        render_slurm_resources(cfg, "es")
        es["pdanalysis_dir"] = st.text_input("PDAnalysis directory", value=es.get("pdanalysis_dir", ""))

        st.markdown("**Reference structure** (set exactly one)")
        es["ref_dir"] = st.text_input("Reference dir (multi-run)", value=es.get("ref_dir", ""))
        es["ref_path"] = st.text_input("Reference CIF file", value=es.get("ref_path", ""))
        es["ref_seq"] = st.text_input("Reference sequence name", value=es.get("ref_seq", ""))

        c1, c2 = st.columns(2)
        es["min_plddt"] = c1.number_input("Min pLDDT", value=es.get("min_plddt", 70), min_value=0, max_value=100)
        method_str = ", ".join(es.get("method", ["strain"])) if isinstance(es.get("method"), list) else str(es.get("method", "strain"))
        method_input = c2.text_input("Methods (comma-separated)", value=method_str)
        es["method"] = [m.strip() for m in method_input.split(",") if m.strip()]

        cutoffs = es.get("lddt_cutoffs", [0.125, 0.25, 0.5, 1])
        cutoffs_str = ", ".join(str(c) for c in cutoffs)
        cutoffs_input = st.text_input("LDDT cutoffs (comma-separated)", value=cutoffs_str)
        try:
            es["lddt_cutoffs"] = [float(x.strip()) for x in cutoffs_input.split(",") if x.strip()]
        except ValueError:
            st.error("Invalid cutoff values")

        es["env_path"] = st.text_input("ES env path", value=es.get("env_path", ""))
        cfg["es"] = es

    with st.expander("SLURM Settings"):
        slurm = cfg.get("slurm", {})
        slurm["log_dir"] = st.text_input("Log directory", value=slurm.get("log_dir", ""))
        c1, c2 = st.columns(2)
        slurm["partition"] = c1.text_input("Default partition", value=slurm.get("partition", ""))
        slurm["account"] = c2.text_input("Account", value=slurm.get("account", ""))
        slurm["email"] = st.text_input("Email", value=slurm.get("email", ""))

        st.markdown("**Per-stage partition overrides** (leave empty for default)")
        for stage in ["msa", "boltz", "esm", "esmfold", "es"]:
            override = slurm.get(stage, {})
            val = st.text_input(f"{stage} partition", value=override.get("partition", ""), key=f"slurm_{stage}")
            if val:
                # Preserve any non-partition keys that might already be set (e.g. resources)
                preserved = {k: v for k, v in override.items() if k != "partition"}
                slurm[stage] = {**preserved, "partition": val}
            else:
                # Drop the stage block entirely if it had only a partition key
                if override and set(override.keys()) <= {"partition"}:
                    slurm.pop(stage, None)
                elif "partition" in override:
                    override.pop("partition", None)
                    slurm[stage] = override

        # Resource overrides written by the estimator's "Apply" button
        resources = slurm.get("resources", {})
        if resources:
            st.markdown("**Per-stage resource overrides** (set by Apply estimates)")
            for stage in ["msa", "boltz", "esm", "esmfold", "es"]:
                r = resources.get(stage)
                if not r:
                    continue
                st.caption(
                    f"`{stage}`: mem={r.get('mem_mb', '?')}MB, "
                    f"runtime={r.get('runtime', '?')}min, "
                    f"cpus={r.get('cpus_per_task', '?')}, "
                    f"gpus={r.get('gpus', '?')}"
                )
            if st.button("Clear resource overrides", key="clear_slurm_resources"):
                slurm.pop("resources", None)
                cfg["slurm"] = slurm
                save_config(session, cfg)
                st.rerun()
        cfg["slurm"] = slurm

    st.divider()
    if st.button("Save Configuration", type="primary", width="stretch"):
        save_config(session, cfg)
        st.success("Config saved! (backup in session directory)")

    with st.expander("View raw YAML"):
        st.code(yaml.dump(cfg, default_flow_style=False, sort_keys=False), language="yaml")


# =========================================================================
# TAB 2: Run Pipeline
# =========================================================================
with tab_run:
    cfg = load_config(session)
    pipeline = cfg.get("pipeline", {})

    st.subheader("Pipeline Summary")
    stages = ["msa", "boltz", "esm", "esmfold", "es"]
    active = [s.upper() for s in stages if pipeline.get(s, False)]
    if active:
        st.info(f"Active stages: {' → '.join(active)}")
    else:
        st.warning("No stages enabled. Go to Configuration tab to enable stages.")

    inp = cfg.get("input", {})
    out = cfg.get("output", {})
    st.markdown(f"**Input:** `{inp.get('fasta_dir', 'not set')}`")
    st.markdown(f"**Output:** `{out.get('parent_dir', 'not set')}`")

    # Snakemake process status + elapsed clock
    running, pid, start_time = snakemake_status(session)
    if running:
        elapsed = format_elapsed(start_time) if start_time else "unknown"
        st.success(f"Snakemake is running (PID {pid}) — elapsed: {elapsed}")
        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("View log tail"):
                if session.log_file.exists():
                    st.code(read_log_tail(session.log_file, 50), language="text")
        with c2:
            if st.button("Stop Pipeline", type="secondary"):
                if stop_snakemake(session):
                    st.warning("Snakemake stopped. Already-submitted SLURM jobs will continue running.")
                    st.rerun()
                else:
                    st.error("Could not stop snakemake process.")
    else:
        if pid is not None:
            msg = f"Last snakemake run finished (was PID {pid})"
            if start_time:
                msg += f" — started at {start_time}"
            st.info(msg)
            if session.log_file.exists() and st.button("View last run log"):
                st.code(read_log_tail(session.log_file, 80), language="text")

    st.divider()

    # Dry run
    st.subheader("Dry Run")
    if st.button("Run snakemake -n (dry run)", width="stretch"):
        with st.spinner("Running dry run..."):
            output = run_cmd(
                ["snakemake", "--profile", "profiles/slurm/",
                 "--configfile", str(session.config_path), "-n"],
                timeout=60,
            )
        if output.strip():
            st.code(output, language="text")
        else:
            st.success("Nothing to do — all outputs are up to date.")

    st.divider()

    # Launch / Rerun buttons
    st.subheader("Launch Pipeline")
    launch_errors = validate_launch_inputs(cfg)
    if launch_errors:
        st.error("Launch blocked by input validation:")
        for err in launch_errors:
            st.text(f"- {err}")
    if running:
        st.warning("Snakemake is already running. Stop it above before launching a new run.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            confirm = st.checkbox("I understand, submit SLURM jobs")
            if st.button(
                "Launch",
                type="primary",
                disabled=(not confirm) or bool(launch_errors),
                width="stretch",
            ):
                new_pid = launch_snakemake(session)
                st.success(f"Snakemake launched in background (PID {new_pid}).")
                st.rerun()
        with c2:
            confirm_rerun = st.checkbox("Resume incomplete jobs")
            if st.button(
                "Rerun Incomplete",
                disabled=(not confirm_rerun) or bool(launch_errors),
                width="stretch",
            ):
                new_pid = launch_snakemake(session, ["--rerun-incomplete"])
                st.success(f"Snakemake --rerun-incomplete launched (PID {new_pid}).")
                st.rerun()


# =========================================================================
# TAB 3: Job Monitor (auto-refreshing)
# =========================================================================
with tab_monitor:
    cfg = load_config(session)
    output_dir = cfg.get("output", {}).get("parent_dir", "")

    # --- Execution clock ---
    running, pid, start_time = snakemake_status(session)
    if running and start_time:
        elapsed = format_elapsed(start_time)
        st.metric("Pipeline running", elapsed)

    # --- Stage progress bars ---
    st.subheader("Pipeline Progress")
    progress = get_stage_progress(cfg)

    if not progress:
        st.info("No stages enabled or output directory not set.")
    else:
        sentinels = {"MSA": ".msa_complete", "Boltz": ".boltz_complete", "ESM": ".esm_complete", "ESMFold": ".esmfold_complete", "ES": "es/.done"}

        for stage, (done, total) in progress.items():
            sentinel = Path(output_dir) / sentinels[stage] if output_dir else None
            is_complete = sentinel and sentinel.exists()

            col_label, col_bar, col_count = st.columns([1, 4, 1])
            with col_label:
                if is_complete:
                    st.markdown(f"**{stage}** :green[done]")
                elif done > 0:
                    st.markdown(f"**{stage}** :orange[running]")
                else:
                    st.markdown(f"**{stage}**")
            with col_bar:
                frac = done / total if total > 0 else 0.0
                st.progress(frac)
            with col_count:
                st.caption(f"{done} / {total}")

        # Auto-refresh based on fastest incomplete stage
        interval = fastest_refresh(progress)
        if interval:
            st.caption(f"Auto-refreshing every {interval}s")
            st_autorefresh(interval=interval * 1000, key="monitor_refresh")

    # --- Active SLURM jobs ---
    st.divider()
    st.subheader("Active SLURM Jobs")
    all_jobs = get_slurm_jobs()
    show_all = st.toggle("Show all user jobs (not just ProtForge)", value=False)

    pf_jobs = [j for j in all_jobs if is_protforge_job(j.get("name", ""))]
    jobs = all_jobs if show_all else pf_jobs

    if not jobs:
        if all_jobs and not pf_jobs:
            st.info(f"No ProtForge jobs running ({len(all_jobs)} other jobs active).")
        else:
            st.info("No SLURM jobs running.")
    else:
        # Summary metrics
        states = {}
        for j in jobs:
            state = j.get("job_state", ["UNKNOWN"])
            state = state[0] if isinstance(state, list) else state
            states[state] = states.get(state, 0) + 1

        state_items = sorted(states.items())
        n_cols = min(len(state_items) + 1, 8)
        cols = st.columns(n_cols)
        cols[0].metric("Total", len(jobs))
        for i, (state, count) in enumerate(state_items[: n_cols - 1]):
            cols[i + 1].metric(state, count)

        # Build job rows with stage labels
        rows = []
        for j in jobs:
            state = j.get("job_state", ["?"])
            state = state[0] if isinstance(state, list) else state
            name = j.get("name", "")
            rows.append({
                "Job ID": j.get("job_id", ""),
                "Stage": job_to_stage(name),
                "Rule": name,
                "State": state,
                "Partition": j.get("partition", ""),
                "Nodes": j.get("nodes", ""),
            })

        st.dataframe(rows, width="stretch", hide_index=True)

    # --- Recent job history (including failed) ---
    st.divider()
    st.subheader("Recent Job History")
    st.caption("Includes completed, failed, and cancelled jobs (last 24h)")

    recent = get_recent_jobs(hours=24)
    pf_recent = [j for j in recent if is_protforge_job(j["name"])]
    show_all_hist = st.toggle("Show all user jobs", value=False, key="hist_all")
    history = recent if show_all_hist else pf_recent

    if not history:
        st.info("No recent jobs found.")
    else:
        # Highlight failures
        hist_rows = []
        all_job_ids = []
        for j in history:
            state = j["state"]
            hist_rows.append({
                "Job ID": j["job_id"],
                "Stage": job_to_stage(j["name"]),
                "Rule": j["name"],
                "State": state,
                "Exit": j["exit_code"],
                "Elapsed": j["elapsed"],
                "Reason": j["reason"],
            })
            all_job_ids.append(j["job_id"])

        st.dataframe(hist_rows, width="stretch", hide_index=True)

        # --- Job log viewer ---
        st.divider()
        st.subheader("Job Log Viewer")
        selected_job = st.selectbox(
            "Select a job to view its log",
            options=all_job_ids,
            format_func=lambda jid: next(
                (f"{jid} — {r['Stage']} ({r['Rule']}) [{r['State']}]" for r in hist_rows if r['Job ID'] == jid),
                str(jid),
            ),
        )
        if selected_job and st.button("Load log"):
            log_path = get_job_log_path(selected_job, cfg)
            if log_path:
                st.caption(f"Reading: {log_path}")
                st.code(read_log_tail(log_path, 150), language="text")
            else:
                st.warning(f"No log file found for job {selected_job}. Check .snakemake/slurm_logs/ or your SLURM log directory.")

    # --- Snakemake controller log ---
    if session.log_file.exists():
        st.divider()
        st.subheader("Snakemake Controller Log")
        if st.button("Show snakemake log tail"):
            st.code(read_log_tail(session.log_file, 100), language="text")
