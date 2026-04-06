"""
ProtForge Web UI — Minimal test app.

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

st.set_page_config(page_title="ProtForge", layout="wide")

st.title("ProtForge")
st.markdown("Pipeline control panel for protein structure prediction.")

# --- Connection info ---
st.header("Connection Info")
col1, col2 = st.columns(2)
with col1:
    st.metric("Host", socket.gethostname())
with col2:
    st.metric("User", os.environ.get("USER", "unknown"))

# --- Quick cluster check ---
st.header("Cluster Status")

# Try squeue to verify we're on a SLURM cluster
try:
    result = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "--noheader", "-h"],
        capture_output=True, text=True, timeout=10,
    )
    job_count = len([l for l in result.stdout.strip().splitlines() if l.strip()])
    st.metric("Your running SLURM jobs", job_count)
except FileNotFoundError:
    st.info("squeue not found — not on a SLURM cluster (or SLURM not in PATH).")
except subprocess.TimeoutExpired:
    st.warning("squeue timed out.")

# --- Config check ---
st.header("Config")
config_path = Path("config.yaml")
if config_path.exists():
    st.success(f"config.yaml found ({config_path.resolve()})")
    with st.expander("View raw config"):
        st.code(config_path.read_text(), language="yaml")
else:
    st.warning("No config.yaml found in current directory.")

st.divider()
st.caption("If you can see this page, the connection is working.")
