"""
ProtForge Web UI — multi-session support.

Usage (on cluster login node):
    streamlit run webapp/app.py --server.port 8501 --server.address 127.0.0.1

Then on your laptop:
    ssh -L 8501:localhost:8501 <user>@<cluster>

Open http://localhost:8501 in your browser.

This module is the thin entrypoint: bootstrap, page config, the session
sidebar, and tab dispatch. The bulk lives in sibling modules:
  pipeline_ops.py  — config I/O, snakemake launch/status (pure)
  monitoring.py    — squeue/sacct queries, output progress (pure)
  ui_helpers.py    — reusable Configuration-tab widgets
  results.py       — output-tree + benchmark readers for the Results tab (pure)
  satmut.py        — saturation-mutagenesis launch/derive/load (pure core)
  config_tab.py / run_tab.py / monitor_tab.py / results_tab.py /
  satmut_tab.py    — tab bodies
"""

import os
import socket

import streamlit as st

from session import (
    migrate_legacy,
    load_registry,
    create_session,
    delete_session,
    foreign_session_owner,
    rename_session,
    list_sessions,
    get_session,
    get_active_session_id,
    set_active_session,
)
from pipeline_ops import load_config, snakemake_status
from config_tab import render_config_tab
from run_tab import render_run_tab
from monitor_tab import render_monitor_tab
from results_tab import render_results_tab
from satmut_tab import render_satmut_tab

USER = os.environ.get("USER", "unknown")
HOST = socket.gethostname()

# Labels for the "start config from" picker — compared by value, so keep them
# distinct from any session name a user might type.
CLUSTER_DEFAULTS = "Cluster defaults (config.kempner.template.yaml)"
EMPTY_CONFIG = "Empty config"

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
        clone_options = [CLUSTER_DEFAULTS, EMPTY_CONFIG] + [s["name"] for s in sessions]
        clone_choice = st.selectbox("Start config from", options=clone_options, key="clone_source")

        if st.button("Create", key="create_session", width="stretch"):
            if not new_name.strip():
                st.error("Enter a session name.")
            else:
                # None seeds from the shipped cluster template; {} is blank.
                base_config = None if clone_choice == CLUSTER_DEFAULTS else {}
                if clone_choice not in (CLUSTER_DEFAULTS, EMPTY_CONFIG):
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
    # The identity this session would submit under. Shown next to the OS user so
    # a config inherited from someone else is visible before anything is queued.
    _slurm = load_config(session).get("slurm", {}) or {}
    _account = _slurm.get("account") or "no account set"
    _email = _slurm.get("email") or "no email set"
    st.caption(f"{_account} · {_email}")

# A session created by another user — a shared checkout, or a copied workspace
# that carried .sessions/ along with it. Their account and email are still in
# this config, so say so rather than letting it look like the user's own.
_owner = foreign_session_owner(session.id)
if _owner:
    st.warning(
        f"Session **{active_name}** was created by **{_owner}**, not {USER}. "
        "Its SLURM account, email and paths are theirs — check the "
        "Configuration tab, or create a new session from the cluster defaults."
    )

tab_config, tab_run, tab_monitor, tab_results, tab_satmut = st.tabs(
    ["Configuration", "Run Pipeline", "Job Monitor", "Results", "Saturation Mutagenesis"]
)

with tab_config:
    render_config_tab(session)

with tab_run:
    render_run_tab(session)

with tab_monitor:
    render_monitor_tab(session)

with tab_results:
    render_results_tab(session)

with tab_satmut:
    render_satmut_tab(session)
