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
                use_container_width=True,
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

        if st.button("Create", key="create_session", use_container_width=True):
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
    """Full import UI: browse directory, CSV/TSV upload, or random mutations."""
    import_mode = st.radio(
        "Import mode",
        ["Browse directory", "Generate from CSV/TSV", "Random mutations"],
        horizontal=True,
        key="dlg_import_mode",
    )

    # Helper to apply result to session config and close dialog
    def _apply_to_config(input_dir: str, file_type: str):
        cfg = load_config(session)
        if "input" not in cfg:
            cfg["input"] = {}
        if "pipeline" not in cfg:
            cfg["pipeline"] = {}
        if file_type == "fasta":
            cfg["input"]["fasta_dir"] = input_dir
            cfg["input"].pop("yaml_dir", None)
            cfg["pipeline"]["msa"] = True
        else:
            cfg["input"]["yaml_dir"] = input_dir
            cfg["input"].pop("fasta_dir", None)
            cfg["pipeline"]["msa"] = False
        save_config(session, cfg)

    # =============================================================
    # MODE 1: Browse directory
    # =============================================================
    if import_mode == "Browse directory":
        st.caption("Browse your cluster filesystem, validate FASTA or YAML files, and set the input directory.")

        if "browse_path" not in st.session_state:
            st.session_state.browse_path = os.environ.get("HOME", "/")

        col_path, col_go = st.columns([5, 1])
        with col_path:
            typed_path = st.text_input(
                "Directory path",
                value=st.session_state.browse_path,
                key="dlg_dir_path",
            )
        with col_go:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Go", use_container_width=True, key="dlg_go"):
                if Path(typed_path).is_dir():
                    st.session_state.browse_path = typed_path
                    st.rerun()
                else:
                    st.error("Not a valid directory.")

        browse_dir = Path(st.session_state.browse_path)

        # Breadcrumb navigation
        parts = browse_dir.parts
        breadcrumb_cols = st.columns(min(len(parts), 10))
        for i, part in enumerate(parts[: len(breadcrumb_cols)]):
            with breadcrumb_cols[i]:
                label = part if part != "/" else "/"
                if st.button(label, key=f"dlg_bc_{i}", use_container_width=True):
                    target = Path(*parts[: i + 1]) if i > 0 else Path("/")
                    st.session_state.browse_path = str(target)
                    st.rerun()

        # Subdirectory listing
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
                            if st.button(f"📁 {d.name}", key=f"dlg_sd_{d.name}", use_container_width=True):
                                st.session_state.browse_path = str(d)
                                st.rerun()

            st.divider()
            st.markdown(f"**Selected:** `{browse_dir}`")

            if st.button("Scan & Validate", type="primary", use_container_width=True, key="dlg_scan"):
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
                        st.warning("Mixed FASTA and YAML files — choose one type per directory.")

                    if result["fasta_results"]:
                        with st.expander(f"FASTA files ({len(result['fasta_results'])})", expanded=result["invalid_count"] > 0):
                            rows = [{
                                "File": r["filename"],
                                "Status": "Valid" if r["valid"] else "INVALID",
                                "Sequences": r["num_sequences"],
                                "Residues": r["total_residues"],
                                "Errors": "; ".join(r["errors"]) if r["errors"] else "",
                            } for r in result["fasta_results"]]
                            st.dataframe(rows, use_container_width=True, hide_index=True)

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
                            st.dataframe(rows, use_container_width=True, hide_index=True)

                    st.divider()
                    if result["valid_count"] > 0 and result["file_type"] in ("fasta", "yaml"):
                        if result["invalid_count"] > 0:
                            st.warning(
                                f"{result['invalid_count']} invalid file(s) found. "
                                "Only valid files will be copied."
                            )
                        apply_label = (
                            f"Apply as {'FASTA' if result['file_type'] == 'fasta' else 'YAML'} "
                            f"input ({result['valid_count']} valid files)"
                        )
                        if st.button(apply_label, type="primary", use_container_width=True, key="dlg_apply_dir"):
                            clean_dir = session.dir / f"input_{result['file_type']}"
                            if clean_dir.exists():
                                import shutil as _shutil
                                _shutil.rmtree(clean_dir)
                            copy_valid_files(result, clean_dir)
                            _apply_to_config(str(clean_dir), result["file_type"])
                            st.success("Config updated!")
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
                    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

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

                    can_gen = csv_mode == "sequences" or (csv_mode == "mutations" and ref_sequence)
                    if st.button("Generate files", type="primary", disabled=not can_gen, use_container_width=True, key="dlg_gen_csv"):
                        output_dir = session.dir / f"generated_{file_type}"
                        if output_dir.exists():
                            import shutil as _shutil
                            _shutil.rmtree(output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)

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
                            _apply_to_config(str(output_dir), file_type)
                            st.success(f"Generated {generated} files. Config updated!")
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

                if st.button("Generate random mutants", type="primary", use_container_width=True, key="dlg_gen_rand"):
                    import random as _random
                    _random.seed(seed)

                    aa_list = list(_VA)
                    seq_len = len(ref_seq_rand)
                    output_dir = session.dir / f"random_mutants_{file_type_rand}"
                    if output_dir.exists():
                        import shutil as _shutil
                        _shutil.rmtree(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)

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

                    _apply_to_config(str(output_dir), file_type_rand)
                    st.success(f"Generated {generated} mutants. Config updated!")

# =========================================================================
# TAB 1: Configuration
# =========================================================================
with tab_config:
    cfg = load_config(session)

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

        col_fasta, col_import = st.columns([5, 1])
        with col_fasta:
            inp["fasta_dir"] = st.text_input("FASTA directory", value=inp.get("fasta_dir", ""))
        with col_import:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Import", use_container_width=True, key="open_import"):
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
    if st.button("Save Configuration", type="primary", use_container_width=True):
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
    if st.button("Run snakemake -n (dry run)", use_container_width=True):
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
    if running:
        st.warning("Snakemake is already running. Stop it above before launching a new run.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            confirm = st.checkbox("I understand, submit SLURM jobs")
            if st.button("Launch", type="primary", disabled=not confirm, use_container_width=True):
                new_pid = launch_snakemake(session)
                st.success(f"Snakemake launched in background (PID {new_pid}).")
                st.rerun()
        with c2:
            confirm_rerun = st.checkbox("Resume incomplete jobs")
            if st.button("Rerun Incomplete", disabled=not confirm_rerun, use_container_width=True):
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

        st.dataframe(rows, use_container_width=True, hide_index=True)

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

        st.dataframe(hist_rows, use_container_width=True, hide_index=True)

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
