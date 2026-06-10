"""
Run Pipeline tab for the ProtForge web UI.

Shows the active-stage summary, controller status, a dry-run button, and the
Launch / Rerun-incomplete controls (gated on input validation).
"""

import streamlit as st

from session import Session
from pipeline_ops import (
    load_config,
    run_cmd,
    launch_snakemake,
    validate_launch_inputs,
    snakemake_status,
    stop_snakemake,
)
from monitoring import format_elapsed, read_log_tail


def render_run_tab(session: Session):
    cfg = load_config(session)
    pipeline = cfg.get("pipeline", {})

    st.subheader("Pipeline Summary")
    _labels = {"msa": "MSA", "boltz": "Boltz", "openfold": "OpenFold3",
               "esmc": "ESMC", "esmfold": "ESMFold2"}
    active = [_labels[s] for s in ["msa", "boltz", "openfold", "esmc", "esmfold"]
              if pipeline.get(s, False)]
    if cfg.get("esmc", {}).get("sae", {}).get("enabled"):
        active.append("ESMC-SAE")
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
