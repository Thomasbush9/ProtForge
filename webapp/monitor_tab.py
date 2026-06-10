"""
Job Monitor tab for the ProtForge web UI (auto-refreshing).

Renders per-stage progress bars from output artifacts, active SLURM jobs,
recent job history (incl. failures), and a per-job log viewer.
"""

from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from session import Session
from pipeline_ops import load_config, snakemake_status
from monitoring import (
    format_elapsed,
    get_stage_progress,
    fastest_refresh,
    get_slurm_jobs,
    get_recent_jobs,
    is_protforge_job,
    job_to_stage,
    get_job_log_path,
    read_log_tail,
)


def render_monitor_tab(session: Session):
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
        def _stage_complete(stage: str) -> bool:
            """ESMC / ESMC-SAE have one sentinel per model size — complete only
            when every configured size is done."""
            if not output_dir:
                return False
            base = Path(output_dir)
            simple = {"MSA": ".msa_complete", "Boltz": ".boltz_complete",
                      "ESMFold": ".esmfold_complete"}
            if stage in simple:
                return (base / simple[stage]).exists()
            esmc = cfg.get("esmc", {})
            if stage == "ESMC":
                sizes = esmc.get("models", []) or []
                return bool(sizes) and all((base / f".esmc_{s}_complete").exists() for s in sizes)
            if stage == "ESMC-SAE":
                sizes = esmc.get("sae", {}).get("sizes") or esmc.get("models", []) or []
                return bool(sizes) and all((base / f".esmc_sae_{s}_complete").exists() for s in sizes)
            return False

        for stage, (done, total) in progress.items():
            is_complete = _stage_complete(stage)

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
                # Clamp: done can briefly exceed total (extra artifacts, reruns)
                # and st.progress rejects values outside [0, 1].
                frac = max(0.0, min(1.0, frac))
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
