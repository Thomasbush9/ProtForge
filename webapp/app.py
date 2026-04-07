"""
ProtForge Web UI.

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

# ---------------------------------------------------------------------------
# Globals — resolve paths relative to the repo root, not cwd
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RUN_META_FILE = REPO_ROOT / ".snakemake_run.json"
LOG_FILE = REPO_ROOT / "snakemake_run.log"
USER = os.environ.get("USER", "unknown")
HOST = socket.gethostname()

# Per-stage auto-refresh intervals (seconds)
REFRESH_INTERVALS = {"MSA": 300, "Boltz": 60, "ESM": 10, "ES": 10}

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
    "collect_es_paths": "ES",
    "run_es_all": "ES",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return {}


def save_config(cfg: dict):
    if CONFIG_PATH.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = REPO_ROOT / ".config_backups"
        backup_dir.mkdir(exist_ok=True)
        shutil.copy2(CONFIG_PATH, backup_dir / f"config_{ts}.yaml")
    CONFIG_PATH.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


def run_cmd(cmd: list[str], timeout: int = 30) -> str:
    """Run a short-lived command and return stdout+stderr."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT)
        return r.stdout + r.stderr
    except FileNotFoundError:
        return f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return "Command timed out."


def launch_snakemake(extra_args: list[str] | None = None):
    """Launch snakemake as a background process with metadata tracking."""
    cmd = ["snakemake", "--profile", "profiles/slurm/"]
    if extra_args:
        cmd.extend(extra_args)
    with open(LOG_FILE, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=REPO_ROOT)
    # Write metadata for robust process identification
    meta = {
        "pid": proc.pid,
        "start_time": datetime.now().isoformat(),
        "command": " ".join(cmd),
    }
    RUN_META_FILE.write_text(json.dumps(meta))
    return proc.pid


def snakemake_status() -> tuple[bool, int | None, str | None]:
    """Check if a snakemake process is running. Returns (running, pid, start_time)."""
    if not RUN_META_FILE.exists():
        return False, None, None
    try:
        meta = json.loads(RUN_META_FILE.read_text())
    except (json.JSONDecodeError, ValueError):
        return False, None, None
    pid = meta.get("pid")
    start_time = meta.get("start_time")
    if pid is None:
        return False, None, None
    try:
        os.kill(pid, 0)
        # Verify it's actually snakemake, not a recycled PID
        r = subprocess.run(["ps", "-p", str(pid), "-o", "comm="], capture_output=True, text=True, timeout=5)
        if "snakemake" not in r.stdout.lower() and "python" not in r.stdout.lower():
            RUN_META_FILE.unlink(missing_ok=True)
            return False, pid, start_time
        return True, pid, start_time
    except (OSError, subprocess.TimeoutExpired):
        RUN_META_FILE.unlink(missing_ok=True)
        return False, pid, start_time


def stop_snakemake() -> bool:
    """Stop the running snakemake process. Returns True if stopped."""
    if not RUN_META_FILE.exists():
        return False
    try:
        meta = json.loads(RUN_META_FILE.read_text())
        pid = meta.get("pid")
        if pid:
            import signal
            os.kill(pid, signal.SIGTERM)
            RUN_META_FILE.unlink(missing_ok=True)
            return True
    except (OSError, json.JSONDecodeError):
        pass
    RUN_META_FILE.unlink(missing_ok=True)
    return False


def format_elapsed(start_iso: str) -> str:
    """Format elapsed time since ISO timestamp as 'Xh Ym Zs'."""
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
    """Get current user's SLURM jobs (running/pending) as list of dicts."""
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
    """Get recent jobs (including completed/failed) via sacct."""
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
            # Skip sub-steps like "12345.batch" or "12345.extern"
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
    """Check if a job name matches a ProtForge/Snakemake rule."""
    return any(rule in job_name for rule in RULE_TO_STAGE)


def job_to_stage(job_name: str) -> str:
    """Map a SLURM job name to a pipeline stage."""
    for rule, stage in RULE_TO_STAGE.items():
        if rule in job_name:
            return stage
    return "Other"


def get_job_log_path(job_id: int | str) -> Path | None:
    """Find the SLURM log file for a job ID."""
    slurm_logs = REPO_ROOT / ".snakemake" / "slurm_logs"
    if slurm_logs.exists():
        for log in slurm_logs.rglob(f"{job_id}.log"):
            return log
    cfg = load_config()
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
    """Read the last n lines of a log file."""
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
    """Count expected sequences from input sources."""
    output_dir = Path(cfg.get("output", {}).get("parent_dir", ""))
    seq_dir = output_dir / "sequences"
    fasta_dir = Path(cfg.get("input", {}).get("fasta_dir", ""))
    yaml_dir = Path(cfg.get("input", {}).get("yaml_dir", ""))
    pipeline = cfg.get("pipeline", {})

    # If MSA is off and yaml_dir is set, count YAMLs
    if not pipeline.get("msa") and yaml_dir != Path("") and yaml_dir.is_dir():
        return count_files(yaml_dir, "*.yaml")
    # Otherwise count input FASTAs
    if fasta_dir.is_dir():
        return count_files(fasta_dir, "*.fasta") + count_files(fasta_dir, "*.fa")
    # Fallback: count existing sequence dirs
    if seq_dir.is_dir():
        return len([d for d in seq_dir.iterdir() if d.is_dir()])
    return 0


def get_stage_progress(cfg: dict) -> dict:
    """Count done vs total for each pipeline stage based on output files."""
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
            # Single-run: count sequences with any CIF
            done = 0
            if seq_dir.is_dir():
                for d in seq_dir.iterdir():
                    boltz_dir = d / "boltz"
                    if boltz_dir.is_dir() and list(boltz_dir.glob("*_model_*.cif")):
                        done += 1
            progress["Boltz"] = (done, total)
        else:
            # Multi-run: track completed runs out of total * num_runs
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

    if pipeline.get("es"):
        es_dir = output_dir / "es"
        done = count_files(es_dir, "*.csv")
        progress["ES"] = (done, total)

    return progress


def fastest_refresh(progress: dict) -> int | None:
    """Return the shortest refresh interval among active incomplete stages."""
    intervals = []
    for stage, (done, total) in progress.items():
        if total > 0 and done < total:
            intervals.append(REFRESH_INTERVALS.get(stage, 60))
    return min(intervals) if intervals else None


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ProtForge", layout="wide")

col_title, col_info = st.columns([3, 1])
with col_title:
    st.title("ProtForge")
with col_info:
    st.caption(f"{USER}@{HOST}")

tab_config, tab_run, tab_monitor = st.tabs(["Configuration", "Run Pipeline", "Job Monitor"])

# =========================================================================
# TAB 1: Configuration
# =========================================================================
with tab_config:
    cfg = load_config()

    with st.expander("Pipeline Stages", expanded=True):
        pipeline = cfg.get("pipeline", {})
        cols = st.columns(4)
        pipeline["msa"] = cols[0].toggle("MSA", value=pipeline.get("msa", True))
        pipeline["boltz"] = cols[1].toggle("Boltz", value=pipeline.get("boltz", True))
        pipeline["esm"] = cols[2].toggle("ESM", value=pipeline.get("esm", True))
        pipeline["es"] = cols[3].toggle("ES", value=pipeline.get("es", False))
        cfg["pipeline"] = pipeline

    with st.expander("Input / Output", expanded=True):
        inp = cfg.get("input", {})
        out = cfg.get("output", {})
        inp["fasta_dir"] = st.text_input("FASTA directory", value=inp.get("fasta_dir", ""))
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

    with st.expander("MSA Settings"):
        msa = cfg.get("msa", {})
        msa["max_files_per_job"] = st.number_input("Files per job", value=msa.get("max_files_per_job", 25), min_value=1)
        msa["mmseq2_db"] = st.text_input("MMseqs2 DB", value=msa.get("mmseq2_db", ""))
        msa["colabfold_db"] = st.text_input("ColabFold DB", value=msa.get("colabfold_db", ""))
        msa["colabfold_bin"] = st.text_input("ColabFold bin", value=msa.get("colabfold_bin", ""))
        cfg["msa"] = msa

    with st.expander("Boltz Settings"):
        boltz = cfg.get("boltz", {})
        c1, c2 = st.columns(2)
        boltz["max_files_per_job"] = c1.number_input("Files per job ", value=boltz.get("max_files_per_job", 25), min_value=1)
        boltz["num_runs"] = c2.number_input("Runs per sequence", value=boltz.get("num_runs", 1), min_value=1)
        c1, c2 = st.columns(2)
        boltz["recycling_steps"] = c1.number_input("Recycling steps", value=boltz.get("recycling_steps", 10), min_value=1)
        boltz["diffusion_samples"] = c2.number_input("Diffusion samples", value=boltz.get("diffusion_samples", 25), min_value=1)
        boltz["delete_msa_after_processing"] = st.toggle(
            "Delete MSA after Boltz", value=boltz.get("delete_msa_after_processing", False),
        )
        boltz["cache_dir"] = st.text_input("Boltz cache dir", value=boltz.get("cache_dir", ""))
        boltz["env_path"] = st.text_input("Boltz env path", value=boltz.get("env_path", ""))
        cfg["boltz"] = boltz

    with st.expander("ESM Settings"):
        esm = cfg.get("esm", {})
        esm["num_chunks"] = st.number_input("Chunks", value=esm.get("num_chunks", 1), min_value=1)
        esm["env_path"] = st.text_input("ESM env path", value=esm.get("env_path", ""))
        esm["cache_dir"] = st.text_input("ESM cache dir", value=esm.get("cache_dir", ""))
        cfg["esm"] = esm

    with st.expander("ES Settings"):
        es = cfg.get("es", {})
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
        for stage in ["msa", "boltz", "esm", "es"]:
            override = slurm.get(stage, {})
            val = st.text_input(f"{stage} partition", value=override.get("partition", ""), key=f"slurm_{stage}")
            if val:
                slurm[stage] = {"partition": val}
            else:
                slurm.pop(stage, None)
        cfg["slurm"] = slurm

    st.divider()
    if st.button("Save Configuration", type="primary", width="stretch"):
        save_config(cfg)
        st.success("config.yaml saved! (backup in .config_backups/)")

    with st.expander("View raw YAML"):
        st.code(yaml.dump(cfg, default_flow_style=False, sort_keys=False), language="yaml")


# =========================================================================
# TAB 2: Run Pipeline
# =========================================================================
with tab_run:
    cfg = load_config()
    pipeline = cfg.get("pipeline", {})

    st.subheader("Pipeline Summary")
    stages = ["msa", "boltz", "esm", "es"]
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
    running, pid, start_time = snakemake_status()
    if running:
        elapsed = format_elapsed(start_time) if start_time else "unknown"
        st.success(f"Snakemake is running (PID {pid}) — elapsed: {elapsed}")
        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("View log tail"):
                if LOG_FILE.exists():
                    st.code(read_log_tail(LOG_FILE, 50), language="text")
        with c2:
            if st.button("Stop Pipeline", type="secondary"):
                if stop_snakemake():
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
            if LOG_FILE.exists() and st.button("View last run log"):
                st.code(read_log_tail(LOG_FILE, 80), language="text")

    st.divider()

    # Dry run
    st.subheader("Dry Run")
    if st.button("Run snakemake -n (dry run)", width="stretch"):
        with st.spinner("Running dry run..."):
            output = run_cmd(
                ["snakemake", "--profile", "profiles/slurm/", "-n"],
                timeout=60,
            )
        if output.strip():
            st.code(output, language="text")
        else:
            st.success("Nothing to do — all outputs are up to date.")

    st.divider()

    # Launch / Rerun buttons
    st.subheader("Launch Pipeline")
    if running:
        st.warning("Snakemake is already running. Stop it above before launching a new run.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            confirm = st.checkbox("I understand, submit SLURM jobs")
            if st.button("Launch", type="primary", disabled=not confirm, width="stretch"):
                new_pid = launch_snakemake()
                st.success(f"Snakemake launched in background (PID {new_pid}).")
                st.rerun()
        with c2:
            confirm_rerun = st.checkbox("Resume incomplete jobs")
            if st.button("Rerun Incomplete", disabled=not confirm_rerun, width="stretch"):
                new_pid = launch_snakemake(["--rerun-incomplete"])
                st.success(f"Snakemake --rerun-incomplete launched (PID {new_pid}).")
                st.rerun()


# =========================================================================
# TAB 3: Job Monitor (auto-refreshing)
# =========================================================================
with tab_monitor:
    cfg = load_config()
    output_dir = cfg.get("output", {}).get("parent_dir", "")

    # --- Execution clock ---
    running, pid, start_time = snakemake_status()
    if running and start_time:
        elapsed = format_elapsed(start_time)
        st.metric("Pipeline running", elapsed)

    # --- Stage progress bars ---
    st.subheader("Pipeline Progress")
    progress = get_stage_progress(cfg)

    if not progress:
        st.info("No stages enabled or output directory not set.")
    else:
        sentinels = {"MSA": ".msa_complete", "Boltz": ".boltz_complete", "ESM": ".esm_complete", "ES": "es/.done"}

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
            log_path = get_job_log_path(selected_job)
            if log_path:
                st.caption(f"Reading: {log_path}")
                st.code(read_log_tail(log_path, 150), language="text")
            else:
                st.warning(f"No log file found for job {selected_job}. Check .snakemake/slurm_logs/ or your SLURM log directory.")

    # --- Snakemake controller log ---
    if LOG_FILE.exists():
        st.divider()
        st.subheader("Snakemake Controller Log")
        if st.button("Show snakemake log tail"):
            st.code(read_log_tail(LOG_FILE, 100), language="text")
