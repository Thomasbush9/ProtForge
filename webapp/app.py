"""
ProtForge Web UI.

Usage (on cluster login node):
    streamlit run webapp/app.py --server.port 8501 --server.address 127.0.0.1

Then on your laptop:
    ssh -L 8501:localhost:8501 <user>@<cluster>

Open http://localhost:8501 in your browser.
"""

import os
import subprocess
import socket
from pathlib import Path

import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("config.yaml")
USER = os.environ.get("USER", "unknown")
HOST = socket.gethostname()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return {}


def save_config(cfg: dict):
    CONFIG_PATH.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


def run_cmd(cmd: list[str], timeout: int = 30) -> str:
    """Run a command and return stdout, or error string."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    except FileNotFoundError:
        return f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return "Command timed out."


def get_slurm_jobs() -> list[dict]:
    """Get current user's SLURM jobs as list of dicts."""
    try:
        r = subprocess.run(
            ["squeue", "-u", USER, "--json"],
            capture_output=True, text=True, timeout=15,
        )
        import json
        data = json.loads(r.stdout)
        return data.get("jobs", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ProtForge", layout="wide")

# Header
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

    # --- Pipeline toggles ---
    with st.expander("Pipeline Stages", expanded=True):
        pipeline = cfg.get("pipeline", {})
        cols = st.columns(4)
        pipeline["msa"] = cols[0].toggle("MSA", value=pipeline.get("msa", True))
        pipeline["boltz"] = cols[1].toggle("Boltz", value=pipeline.get("boltz", True))
        pipeline["esm"] = cols[2].toggle("ESM", value=pipeline.get("esm", True))
        pipeline["es"] = cols[3].toggle("ES", value=pipeline.get("es", False))
        cfg["pipeline"] = pipeline

    # --- Input / Output ---
    with st.expander("Input / Output", expanded=True):
        inp = cfg.get("input", {})
        out = cfg.get("output", {})
        inp["fasta_dir"] = st.text_input("FASTA directory", value=inp.get("fasta_dir", ""))
        yaml_dir = inp.get("yaml_dir", "")
        inp["yaml_dir"] = st.text_input(
            "YAML directory (when MSA is off)", value=yaml_dir,
            help="Set this when pipeline.msa is false",
        )
        # Remove yaml_dir if empty so it doesn't clutter config
        if not inp["yaml_dir"]:
            inp.pop("yaml_dir", None)
        out["parent_dir"] = st.text_input("Output directory", value=out.get("parent_dir", ""))
        cfg["input"] = inp
        cfg["output"] = out

    # --- MSA ---
    with st.expander("MSA Settings"):
        msa = cfg.get("msa", {})
        c1, c2 = st.columns(2)
        msa["max_files_per_job"] = c1.number_input("Files per job", value=msa.get("max_files_per_job", 25), min_value=1)
        msa["array_max_concurrency"] = c2.number_input("Max concurrency", value=msa.get("array_max_concurrency", 10), min_value=1)
        msa["mmseq2_db"] = st.text_input("MMseqs2 DB", value=msa.get("mmseq2_db", ""))
        msa["colabfold_db"] = st.text_input("ColabFold DB", value=msa.get("colabfold_db", ""))
        msa["colabfold_bin"] = st.text_input("ColabFold bin", value=msa.get("colabfold_bin", ""))
        cfg["msa"] = msa

    # --- Boltz ---
    with st.expander("Boltz Settings"):
        boltz = cfg.get("boltz", {})
        c1, c2, c3 = st.columns(3)
        boltz["max_files_per_job"] = c1.number_input("Files per job ", value=boltz.get("max_files_per_job", 25), min_value=1)
        boltz["array_max_concurrency"] = c2.number_input("Max concurrency ", value=boltz.get("array_max_concurrency", 10), min_value=1)
        boltz["num_runs"] = c3.number_input("Runs per sequence", value=boltz.get("num_runs", 1), min_value=1)
        c1, c2 = st.columns(2)
        boltz["recycling_steps"] = c1.number_input("Recycling steps", value=boltz.get("recycling_steps", 10), min_value=1)
        boltz["diffusion_samples"] = c2.number_input("Diffusion samples", value=boltz.get("diffusion_samples", 25), min_value=1)
        boltz["delete_msa_after_processing"] = st.toggle(
            "Delete MSA after Boltz", value=boltz.get("delete_msa_after_processing", False),
        )
        boltz["cache_dir"] = st.text_input("Boltz cache dir", value=boltz.get("cache_dir", ""))
        boltz["env_path"] = st.text_input("Boltz env path", value=boltz.get("env_path", ""))
        cfg["boltz"] = boltz

    # --- ESM ---
    with st.expander("ESM Settings"):
        esm = cfg.get("esm", {})
        c1, c2 = st.columns(2)
        esm["num_chunks"] = c1.number_input("Chunks", value=esm.get("num_chunks", 1), min_value=1)
        esm["array_max_concurrency"] = c2.number_input("Max concurrency  ", value=esm.get("array_max_concurrency", 20), min_value=1)
        esm["env_path"] = st.text_input("ESM env path", value=esm.get("env_path", ""))
        esm["cache_dir"] = st.text_input("ESM cache dir", value=esm.get("cache_dir", ""))
        cfg["esm"] = esm

    # --- ES ---
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

    # --- SLURM ---
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

    # --- Save button ---
    st.divider()
    if st.button("Save Configuration", type="primary", use_container_width=True):
        save_config(cfg)
        st.success("config.yaml saved!")

    with st.expander("View raw YAML"):
        st.code(yaml.dump(cfg, default_flow_style=False, sort_keys=False), language="yaml")


# =========================================================================
# TAB 2: Run Pipeline
# =========================================================================
with tab_run:
    cfg = load_config()
    pipeline = cfg.get("pipeline", {})

    # Show what will run
    st.subheader("Pipeline Summary")
    stages = ["msa", "boltz", "esm", "es"]
    active = [s.upper() for s in stages if pipeline.get(s, False)]
    if active:
        st.info(f"Active stages: {' → '.join(active)}")
    else:
        st.warning("No stages enabled. Go to Configuration tab to enable stages.")

    # Input summary
    inp = cfg.get("input", {})
    out = cfg.get("output", {})
    st.markdown(f"**Input:** `{inp.get('fasta_dir', 'not set')}`")
    st.markdown(f"**Output:** `{out.get('parent_dir', 'not set')}`")

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

    # Full run
    st.subheader("Launch Pipeline")
    st.warning("This will submit SLURM jobs to the cluster.")
    confirm = st.checkbox("I understand, launch the pipeline")
    if st.button("Launch", type="primary", disabled=not confirm, use_container_width=True):
        with st.spinner("Launching snakemake..."):
            output = run_cmd(
                ["snakemake", "--profile", "profiles/slurm/"],
                timeout=120,
            )
        st.code(output, language="text")


# =========================================================================
# TAB 3: Job Monitor
# =========================================================================
with tab_monitor:
    if st.button("Refresh", use_container_width=True):
        st.rerun()

    jobs = get_slurm_jobs()

    if not jobs:
        st.info("No SLURM jobs running.")
    else:
        # Summary metrics
        states = {}
        for j in jobs:
            state = j.get("job_state", ["UNKNOWN"])
            state = state[0] if isinstance(state, list) else state
            states[state] = states.get(state, 0) + 1

        cols = st.columns(len(states) + 1)
        cols[0].metric("Total", len(jobs))
        for i, (state, count) in enumerate(sorted(states.items())):
            cols[i + 1].metric(state, count)

        st.divider()

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
                "Time": j.get("time", {}).get("elapsed", 0) if isinstance(j.get("time"), dict) else "",
            })

        st.dataframe(rows, use_container_width=True, hide_index=True)

    # --- Sentinel file status ---
    st.divider()
    st.subheader("Stage Completion")
    cfg = load_config()
    output_dir = cfg.get("output", {}).get("parent_dir", "")
    if output_dir:
        sentinels = {
            "MSA": ".msa_complete",
            "Boltz": ".boltz_complete",
            "ESM": ".esm_complete",
            "ES": "es/.done",
        }
        cols = st.columns(len(sentinels))
        for i, (stage, sentinel) in enumerate(sentinels.items()):
            path = Path(output_dir) / sentinel
            if path.exists():
                cols[i].success(f"{stage}: Complete")
            else:
                cols[i].warning(f"{stage}: Pending")
    else:
        st.caption("Set output directory in Configuration to see stage status.")
