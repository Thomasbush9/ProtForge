"""
Configuration tab for the ProtForge web UI.

Renders the per-stage config expanders plus the Import and Advanced-Boltz
dialogs. All state is read from / written to the active session's config.yaml
via pipeline_ops; the Save button persists the in-memory edits.
"""

import os
from pathlib import Path

import streamlit as st
import yaml

from session import Session, touch_session
from validate import scan_directory, copy_valid_files
from estimator import compute_input_stats
from pipeline_ops import load_config, save_config
from ui_helpers import (
    autoscan_directory,
    _cached_scan,
    render_estimate_panel,
    render_gpu_preference,
    render_chunk_recommendation,
    render_slurm_resources,
    render_binning_controls,
    render_max_concurrent,
    _SLURM_DEFAULTS,
)


# ---------------------------------------------------------------------------
# Import dialog — opens as a popup from the Configuration tab
# ---------------------------------------------------------------------------
@st.dialog("Import Data", width="large")
def import_dialog(session: Session):
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
def boltz_advanced_dialog(session: Session):
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
# Configuration tab body
# =========================================================================
def render_config_tab(session: Session):
    cfg = load_config(session)

    with st.expander("Pipeline Stages", expanded=True):
        pipeline = cfg.get("pipeline", {})
        cols = st.columns(5)
        pipeline["msa"] = cols[0].toggle("MSA", value=pipeline.get("msa", True))
        pipeline["boltz"] = cols[1].toggle("Boltz", value=pipeline.get("boltz", True))
        pipeline["esmc"] = cols[2].toggle("ESMC", value=pipeline.get("esmc", False))
        pipeline["esmfold"] = cols[3].toggle("ESMFold2", value=pipeline.get("esmfold", False))
        pipeline["openfold"] = cols[4].toggle("OpenFold3", value=pipeline.get("openfold", False))
        # Legacy keys from the pre-container pipeline.
        pipeline.pop("es", None)
        pipeline.pop("esm", None)
        cfg["pipeline"] = pipeline
        st.caption("ESMC-SAE is enabled in the ESMC Settings section below "
                   "(it can run independently of the ESMC embedding toggle).")

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
                import_dialog(session)

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
        render_max_concurrent(msa, "msa")
        msa["mmseq2_db"] = st.text_input("MMseqs2 DB", value=msa.get("mmseq2_db", ""))
        msa["colabfold_db"] = st.text_input("ColabFold DB", value=msa.get("colabfold_db", ""))
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
        render_max_concurrent(boltz, "boltz")
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
        boltz["cache_dir"] = st.text_input(
            "Boltz weights dir", value=boltz.get("cache_dir", ""),
            help="Host dir with the Boltz model weights; bound into the container at /weights.")
        boltz.pop("env_path", None)  # legacy conda path — boltz runs in a container now

        adv_count = len(boltz.get("advanced", {}) or {})
        adv_label = (
            f"Advanced Boltz Options ({adv_count} set)"
            if adv_count else "Advanced Boltz Options..."
        )
        if st.button(adv_label, key="open_boltz_adv"):
            cfg["boltz"] = boltz
            save_config(session, cfg)
            boltz_advanced_dialog(session)

        cfg["boltz"] = boltz

    with st.expander("ESMC Settings"):
        esmc = cfg.get("esmc", {})
        cfg.pop("esm", None)  # drop the legacy fair-esm block
        st.caption("ESM-C embeddings. Each selected size runs as its own parallel "
                   "stage; outputs land in sequences/{seq}/esmc/{size}/. "
                   "Needs only the sequence — with MSA off it runs straight from "
                   "the FASTA directory (no MSA/Boltz required).")
        _all_sizes = ["300M", "600M", "6B"]
        _current = [s for s in esmc.get("models", []) if s in _all_sizes]
        esmc["models"] = st.multiselect(
            "Model sizes", options=_all_sizes,
            default=_current or ["600M"],
            help="Each size launches its own SLURM jobs + sentinel.",
            key="esmc_models",
        )
        esmc["max_files_per_job"] = st.number_input(
            "Files per job", value=int(esmc.get("max_files_per_job", 25)), min_value=1,
            help="Sequences per encode job (also the padded batch size).",
            key="esmc_max_files",
        )
        render_chunk_recommendation("esmc")
        render_gpu_preference("esmc")
        esmc["cache_dir"] = st.text_input(
            "HF cache dir (host)", value=esmc.get("cache_dir", ""),
            help="Host HuggingFace cache; must contain hub/models--biohub--ESMC-* "
                 "(and the SAE repos if SAE is enabled). Bound read-only to /models/hf.",
            key="esmc_cache_dir",
        )

        st.markdown("**Per-size SLURM resources**")
        st.caption("`esmc` below is the shared fallback / estimator target; each "
                   "size can override it (writes slurm.resources.esmc_<size>).")
        render_slurm_resources(cfg, "esmc")
        for _size in esmc["models"]:
            st.markdown(f"_ESMC-{_size}_")
            render_slurm_resources(cfg, f"esmc_{_size}", defaults=_SLURM_DEFAULTS["esmc"])
        render_max_concurrent(esmc, "esmc")

        # --- SAE sub-section ---
        st.markdown("---")
        st.markdown("**ESMC-SAE (sparse activations)**")
        sae = esmc.get("sae", {})
        sae["enabled"] = st.toggle(
            "Extract SAE activations",
            value=bool(sae.get("enabled", False)),
            help="Recomputed from the sequence (independent of the embeddings above), "
                 "so it can run on a prior run's YAMLs. "
                 "Outputs -> sequences/{seq}/sae/{size}/{sae_type}/.",
            key="esmc_sae_enabled",
        )
        if sae["enabled"]:
            c1, c2 = st.columns(2)
            _sae_types = ["all-layers", "mlp"]
            _cur_type = sae.get("sae_type", "all-layers")
            sae["sae_type"] = c1.selectbox(
                "SAE type", options=_sae_types,
                index=_sae_types.index(_cur_type) if _cur_type in _sae_types else 0,
                key="esmc_sae_type",
            )
            sae["layers"] = c2.text_input(
                "Layers", value=str(sae.get("layers", "all")),
                help="'all' = every trained layer, or a comma list e.g. 18,36.",
                key="esmc_sae_layers",
            )
            _sae_default = [s for s in (sae.get("sizes") or esmc["models"]) if s in esmc["models"]]
            sae["sizes"] = st.multiselect(
                "Sizes to extract SAE for", options=esmc["models"],
                default=_sae_default or esmc["models"],
                help="Subset of the ESMC model sizes above.",
                key="esmc_sae_sizes",
            )
            sae["max_files_per_job"] = st.number_input(
                "Files per SAE job", value=int(sae.get("max_files_per_job", 25)), min_value=1,
                key="esmc_sae_max_files",
            )
            st.markdown("_SAE SLURM resources_")
            render_slurm_resources(cfg, "esmc_sae")
        esmc["sae"] = sae
        cfg["esmc"] = esmc

    with st.expander("ESMFold2 Settings"):
        esmfold = cfg.get("esmfold", {})
        st.caption("ESMFold2 (biohub 'fast' variant). Needs only the sequence — "
                   "with MSA off it runs straight from the FASTA directory; with "
                   "MSA on it uses the MSA-stage YAMLs. Outputs -> "
                   "sequences/{seq}/esmfold/fast/.")
        esmfold["max_files_per_job"] = st.number_input(
            "Files per job", value=int(esmfold.get("max_files_per_job", 25)), min_value=1,
            help="Sequences folded per job.",
            key="esmfold_max_files",
        )
        render_chunk_recommendation("esmfold")
        render_gpu_preference("esmfold")
        render_slurm_resources(cfg, "esmfold")
        render_max_concurrent(esmfold, "esmfold")
        esmfold["cache_dir"] = st.text_input(
            "HF cache dir (host)", value=esmfold.get("cache_dir", ""),
            help="Host HF cache; must contain hub/models--biohub--ESMFold2-Fast. "
                 "Bound read-only to /models/hf.",
            key="esmfold_cache_dir",
        )
        c1, c2, c3 = st.columns(3)
        esmfold["num_loops"] = c1.number_input(
            "Num loops", value=int(esmfold.get("num_loops", 3)), min_value=1,
            key="esmfold_num_loops")
        esmfold["num_sampling_steps"] = c2.number_input(
            "Sampling steps", value=int(esmfold.get("num_sampling_steps", 50)), min_value=1,
            key="esmfold_num_sampling_steps")
        esmfold["seed"] = c3.number_input(
            "Seed", value=int(esmfold.get("seed", 0)), min_value=0, step=1,
            key="esmfold_seed")
        # Drop legacy ESMFold v1 keys if a pre-container config still carries them.
        for _k in ("input_type", "num_chunks", "array_max_concurrency", "env_path",
                   "max_seq_len", "bin_by_length", "length_threshold",
                   "num_chunks_short", "num_chunks_long", "mem_short_mb", "mem_long_mb",
                   "time_short_min", "time_long_min", "chunk_size_threshold", "chunk_size"):
            esmfold.pop(_k, None)
        cfg["esmfold"] = esmfold

    with st.expander("OpenFold3 Settings"):
        openfold = cfg.get("openfold", {})
        st.caption("OpenFold3 structure prediction. Runs off the MSA-stage YAMLs "
                   "(in parallel with Boltz). Each query JSON is one SLURM job that "
                   "batches its queries across `GPUs per job` GPUs on one node; "
                   "num jobs == num JSONs. Outputs -> sequences/{seq}/openfold/.")

        # --- Batching: num_batches (jobs) wins, else queries-per-JSON ---
        _use_num_batches = st.toggle(
            "Set number of batches (jobs) directly",
            value=openfold.get("num_batches") is not None,
            help="ON: split all sequences into exactly N JSONs/jobs. "
                 "OFF: fixed number of queries per JSON.",
            key="openfold_use_num_batches",
        )
        c1, c2 = st.columns(2)
        if _use_num_batches:
            openfold["num_batches"] = int(c1.number_input(
                "Number of batches (jobs)",
                value=int(openfold.get("num_batches") or 4), min_value=1,
                key="openfold_num_batches"))
            c2.number_input("Queries per JSON", value=int(openfold.get("max_files_per_job", 25)),
                            min_value=1, disabled=True, key="openfold_max_files_disabled")
        else:
            openfold.pop("num_batches", None)
            openfold["max_files_per_job"] = int(c1.number_input(
                "Queries per JSON", value=int(openfold.get("max_files_per_job", 25)),
                min_value=1, key="openfold_max_files"))
        # NOTE: OpenFold3 has no scaling model in the estimator (estimator.ALL_STAGES
        # excludes it), so chunk-size/GPU recommendations aren't available here yet.
        # Set GPUs per job and SLURM resources manually below.

        openfold["gpus_per_job"] = int(c2.slider(
            "GPUs per job", min_value=1, max_value=4,
            value=int(openfold.get("gpus_per_job", 1)),
            help="GPUs requested per job (one node). Sets pl_trainer_args.devices "
                 "in runner.yaml; queries are batched across them by Lightning.",
            key="openfold_gpus_per_job"))

        c1, c2, c3 = st.columns(3)
        openfold["num_diffusion_samples"] = int(c1.number_input(
            "Diffusion samples", value=int(openfold.get("num_diffusion_samples", 5)),
            min_value=1, key="openfold_diffusion_samples"))
        openfold["num_model_seeds"] = int(c2.number_input(
            "Model seeds", value=int(openfold.get("num_model_seeds", 1)),
            min_value=1, key="openfold_model_seeds"))
        openfold["num_workers"] = int(c3.number_input(
            "Data workers", value=int(openfold.get("num_workers", 10)),
            min_value=1, key="openfold_num_workers"))

        # samples_to_save: int top-N or "all" (max = diffusion samples)
        _ds_max = int(openfold["num_diffusion_samples"])
        _current_save = openfold.get("samples_to_save", 1)
        c1, c2 = st.columns(2)
        _save_all = c1.toggle(
            "Save all samples", value=_current_save == "all",
            help="Keep every diffusion sample. Otherwise keep the top-N by "
                 "ranking score.",
            key="openfold_save_all")
        if _save_all:
            openfold["samples_to_save"] = "all"
            c2.number_input("Top N to save", value=_ds_max, min_value=1, max_value=_ds_max,
                            disabled=True, key="openfold_samples_disabled")
        else:
            _n_default = 1 if _current_save == "all" else int(_current_save)
            _n_default = min(max(_n_default, 1), _ds_max)
            openfold["samples_to_save"] = int(c2.number_input(
                "Top N to save", value=_n_default, min_value=1, max_value=_ds_max,
                help=f"Save top-N samples by ranking score (max = {_ds_max}).",
                key="openfold_samples_to_save"))

        c1, c2 = st.columns(2)
        _fmts = ["cif", "pdb", "cif.gz"]
        _cur_fmt = openfold.get("structure_format", "cif")
        openfold["structure_format"] = c1.selectbox(
            "Structure format", options=_fmts,
            index=_fmts.index(_cur_fmt) if _cur_fmt in _fmts else 0,
            key="openfold_structure_format")
        openfold["write_full_confidence"] = c2.toggle(
            "Write full confidence scores",
            value=bool(openfold.get("write_full_confidence", True)),
            key="openfold_write_full_confidence")

        _seeds_str = st.text_input(
            "Explicit seeds (comma list, optional)",
            value=",".join(str(s) for s in (openfold.get("seeds") or [])),
            help="experiment_settings.seeds in runner.yaml. Leave blank to use "
                 "num_model_seeds.",
            key="openfold_seeds")
        _seeds = [int(s) for s in _seeds_str.replace(" ", "").split(",") if s]
        if _seeds:
            openfold["seeds"] = _seeds
        else:
            openfold.pop("seeds", None)

        _use_recycling = st.toggle(
            "Override recycling iterations",
            value=openfold.get("recycling_iters") is not None,
            help="Writes model_update.custom.num_recycling_iters in runner.yaml. "
                 "Leave off to use the predict preset's default.",
            key="openfold_use_recycling")
        if _use_recycling:
            openfold["recycling_iters"] = int(st.number_input(
                "Recycling iterations", value=int(openfold.get("recycling_iters") or 4),
                min_value=1, key="openfold_recycling_iters"))
        else:
            openfold.pop("recycling_iters", None)

        render_slurm_resources(cfg, "openfold")
        render_max_concurrent(openfold, "openfold")

        openfold["cache_dir"] = st.text_input(
            "OpenFold cache dir (host)", value=openfold.get("cache_dir", ""),
            help="Host dir with OpenFold weights + CCD cache. Bound read-only to "
                 "/models/openfold (OPENFOLD_CACHE).",
            key="openfold_cache_dir")

        # Escape hatch: raw YAML deep-merged into runner.yaml.
        _adv_str = st.text_area(
            "Advanced runner.yaml overrides (YAML, optional)",
            value=(yaml.safe_dump(openfold.get("advanced"), sort_keys=False)
                   if openfold.get("advanced") else ""),
            help="Deep-merged into the generated runner.yaml. For settings not "
                 "covered above (e.g. nested model_update.custom keys).",
            key="openfold_advanced")
        if _adv_str.strip():
            try:
                _adv = yaml.safe_load(_adv_str)
                if isinstance(_adv, dict):
                    openfold["advanced"] = _adv
                else:
                    st.warning("Advanced overrides must be a YAML mapping; ignored.")
                    openfold.pop("advanced", None)
            except yaml.YAMLError as e:
                st.warning(f"Invalid YAML in advanced overrides: {e}")
        else:
            openfold.pop("advanced", None)

        cfg["openfold"] = openfold

    with st.expander("Container Images"):
        containers = cfg.get("containers", {})
        _rt_opts = ["auto", "singularity", "apptainer"]
        _cur_rt = containers.get("runtime", "auto")
        containers["runtime"] = st.selectbox(
            "Runtime", options=_rt_opts,
            index=_rt_opts.index(_cur_rt) if _cur_rt in _rt_opts else 0,
            help="Container binary. 'auto' picks singularity if on PATH, else apptainer.",
            key="containers_runtime",
        )
        containers["colabfold"] = st.text_input(
            "MSA image (.sif)", value=containers.get("colabfold", ""), key="containers_colabfold")
        containers["boltz"] = st.text_input(
            "Boltz image (.sif)", value=containers.get("boltz", ""), key="containers_boltz")
        containers["esmc"] = st.text_input(
            "ESM image (.sif) — serves ESMC + SAE", value=containers.get("esmc", ""),
            key="containers_esmc")
        containers["esmfold"] = st.text_input(
            "ESMFold2 image (.sif)", value=containers.get("esmfold", ""), key="containers_esmfold")
        containers["openfold"] = st.text_input(
            "OpenFold3 image (.sif)", value=containers.get("openfold", ""), key="containers_openfold")
        _gpu = st.text_input(
            "Shared fallback image (.sif, optional)", value=containers.get("gpu", ""),
            help="Used for any stage whose own image field is empty.",
            key="containers_gpu")
        if _gpu:
            containers["gpu"] = _gpu
        else:
            containers.pop("gpu", None)
        cfg["containers"] = containers

    with st.expander("SLURM Settings"):
        slurm = cfg.get("slurm", {})
        slurm["log_dir"] = st.text_input("Log directory", value=slurm.get("log_dir", ""))
        c1, c2 = st.columns(2)
        slurm["partition"] = c1.text_input("Default partition", value=slurm.get("partition", ""))
        slurm["account"] = c2.text_input("Account", value=slurm.get("account", ""))
        slurm["email"] = st.text_input("Email", value=slurm.get("email", ""))

        _cap_jobs = st.toggle(
            "Limit total concurrent cluster jobs",
            value=slurm.get("max_concurrent_jobs") is not None,
            help="Global cap across all stages (Snakemake --jobs), overriding the "
                 "profile default (100). Per-stage caps below further restrict "
                 "individual stages.",
            key="slurm_cap_total_jobs",
        )
        if _cap_jobs:
            slurm["max_concurrent_jobs"] = int(st.number_input(
                "Max concurrent jobs (all stages)",
                value=int(slurm.get("max_concurrent_jobs") or 20), min_value=1,
                key="slurm_max_concurrent_jobs"))
        else:
            slurm.pop("max_concurrent_jobs", None)

        st.markdown("**Per-stage partition overrides** (leave empty for default)")
        for stage in ["msa", "boltz", "esmc", "esmc_sae", "esmfold", "openfold"]:
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
            st.markdown("**Per-stage resource overrides** (set by Apply estimates / per-size rows)")
            for stage in sorted(resources.keys()):
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
