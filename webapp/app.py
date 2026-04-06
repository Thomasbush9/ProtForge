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
PID_FILE = REPO_ROOT / ".snakemake_pid"
LOG_FILE = REPO_ROOT / "snakemake_run.log"
USER = os.environ.get("USER", "unknown")
HOST = socket.gethostname()

# Per-stage auto-refresh intervals (seconds)
REFRESH_INTERVALS = {"MSA": 300, "Boltz": 60, "ESM": 10, "ES": 10}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return {}


def save_config(cfg: dict):
    # Backup before overwriting
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
    """Launch snakemake as a background process with PID tracking."""
    cmd = ["snakemake", "--profile", "profiles/slurm/"]
    if extra_args:
        cmd.extend(extra_args)
    with open(LOG_FILE, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=REPO_ROOT)
    PID_FILE.write_text(str(proc.pid))
    return proc.pid


def snakemake_status() -> tuple[bool, int | None]:
    """Check if a snakemake process is running. Returns (running, pid)."""
    if not PID_FILE.exists():
        return False, None
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, 0)  # signal 0 = check if alive
        return True, pid
    except OSError:
        PID_FILE.unlink(missing_ok=True)
        return False, pid


def get_slurm_jobs() -> list[dict]:
    """Get current user's SLURM jobs as list of dicts."""
    try:
        r = subprocess.run(
            ["squeue", "-u", USER, "--json"],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(r.stdout)
        return data.get("jobs", [])
    except Exception:
        return []


def count_files(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def get_stage_progress(cfg: dict) -> dict:
    """Count done vs total for each pipeline stage based on output files."""
    output_dir = Path(cfg.get("output", {}).get("parent_dir", ""))
    seq_dir = output_dir / "sequences"
    fasta_dir = Path(cfg.get("input", {}).get("fasta_dir", ""))
    pipeline = cfg.get("pipeline", {})

    if fasta_dir.is_dir():
        total = count_files(fasta_dir, "*.fasta") + count_files(fasta_dir, "*.fa")
    elif seq_dir.is_dir():
        total = len([d for d in seq_dir.iterdir() if d.is_dir()])
    else:
        total = 0

    progress = {}

    if pipeline.get("msa"):
        done = count_files(seq_dir, "*/msa/*.a3m")
        progress["MSA"] = (done, total)

    if pipeline.get("boltz"):
        done = 0
        if seq_dir.is_dir():
            for d in seq_dir.iterdir():
                boltz_dir = d / "boltz"
                if boltz_dir.is_dir():
                    if list(boltz_dir.glob("*_model_*.cif")) or list(boltz_dir.glob("run_*/*_model_*.cif")):
                        done += 1
        progress["Boltz"] = (done, total)

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
        c1, c2, c3 = st.columns(3)
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
    if st.button("Save Configuration", type="primary", use_container_width=True):
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

    # Snakemake process status
    running, pid = snakemake_status()
    if running:
        st.success(f"Snakemake is running (PID {pid})")
        if st.button("View log tail"):
            if LOG_FILE.exists():
                lines = LOG_FILE.read_text().splitlines()
                st.code("\n".join(lines[-50:]), language="text")
    else:
        if pid is not None:
            st.info(f"Last snakemake run finished (was PID {pid}).")
            if LOG_FILE.exists() and st.button("View last run log"):
                lines = LOG_FILE.read_text().splitlines()
                st.code("\n".join(lines[-80:]), language="text")

    st.divider()

    # Dry run
    st.subheader("Dry Run")
    if st.button("Run snakemake -n (dry run)", use_container_width=True):
        with st.spinner("Running dry run..."):
            output = run_cmd(
                ["snakemake", "--profile", "profiles/slurm/", "-n", "--quiet"],
                timeout=60,
            )
        st.code(output, language="text")

    st.divider()

    # Launch / Rerun buttons
    st.subheader("Launch Pipeline")
    if running:
        st.warning("Snakemake is already running. Wait for it to finish or stop it manually.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            confirm = st.checkbox("I understand, submit SLURM jobs")
            if st.button("Launch", type="primary", disabled=not confirm, use_container_width=True):
                pid = launch_snakemake()
                st.success(f"Snakemake launched in background (PID {pid}). Check log: {LOG_FILE}")
                st.rerun()
        with c2:
            confirm_rerun = st.checkbox("Resume incomplete jobs")
            if st.button("Rerun Incomplete", disabled=not confirm_rerun, use_container_width=True):
                pid = launch_snakemake(["--rerun-incomplete"])
                st.success(f"Snakemake --rerun-incomplete launched (PID {pid}).")
                st.rerun()


# =========================================================================
# TAB 3: Job Monitor (auto-refreshing)
# =========================================================================
with tab_monitor:
    cfg = load_config()
    output_dir = cfg.get("output", {}).get("parent_dir", "")

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

    # --- SLURM jobs ---
    st.divider()
    st.subheader("SLURM Jobs")
    jobs = get_slurm_jobs()

    if not jobs:
        st.info("No SLURM jobs running.")
    else:
        # Summary metrics — safe for any number of states
        states = {}
        for j in jobs:
            state = j.get("job_state", ["UNKNOWN"])
            state = state[0] if isinstance(state, list) else state
            states[state] = states.get(state, 0) + 1

        state_items = sorted(states.items())
        cols = st.columns(min(len(state_items) + 1, 8))
        cols[0].metric("Total", len(jobs))
        for i, (state, count) in enumerate(state_items[:7]):  # cap at 7 states
            cols[i + 1].metric(state, count)

        # Job table
        rows = []
        for j in jobs:
            state = j.get("job_state", ["?"])
            state = state[0] if isinstance(state, list) else state
            rows.append({
                "Job ID": j.get("job_id", ""),
                "Name": j.get("name", ""),
                "State": state,
                "Partition": j.get("partition", ""),
                "Nodes": j.get("nodes", ""),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
