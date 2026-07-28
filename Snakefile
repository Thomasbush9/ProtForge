"""
ProtForge Snakemake Workflow
============================
Orchestrates the protein prediction pipeline:
  MSA -> Boltz -> ESM / ESMFold

Usage:
  snakemake --profile profiles/slurm/           # Full pipeline via SLURM
  snakemake --profile profiles/slurm/ -n        # Dry run
  snakemake --profile profiles/slurm/ --dag | dot -Tpng > dag.png
  snakemake --profile profiles/slurm/ --rerun-incomplete
"""

import os as _os
import sys as _sys
import time as _time
from pathlib import Path as _Path

# Make workflow.scripts importable for any rule helpers we factor out.
_sys.path.insert(0, str(_Path(workflow.basedir) / "workflow" / "scripts"))
from snake_helpers import check_stage_assets as _check_stage_assets
from snake_helpers import expand_config as _expand_config
from snake_helpers import slurm_identity_errors as _slurm_identity_errors
from snake_helpers import stage_resource as _stage_resource_impl
from snake_helpers import stage_uses_gpu as _stage_uses_gpu_impl
from binning import chunk_resource as _chunk_resource_impl

configfile: "config.yaml"

# Resolve ${PROTFORGE_ROOT}-style placeholders so the shipped cluster templates
# can point at the caller's workspace. Plain absolute paths are unaffected.
config.update(_expand_config(dict(config)))

# Refuse to submit under a SLURM identity copied from the shipped template — an
# unedited `<YOUR_EMAIL>` would otherwise reach sbatch. A *missing* account is
# allowed here: it is legitimate for a local, non-SLURM run.
_identity_errors = _slurm_identity_errors(config, require_account=False)
if _identity_errors:
    raise WorkflowError(
        "Edit your config before running:\n  - " + "\n  - ".join(_identity_errors)
    )

# Refuse to plan a run whose images or weight caches do not exist. Without this
# a dry run of a half-finished install reports a clean DAG and exits 0 — the
# webapp's preflight caught it, `snakemake -n` did not. Set PROTFORGE_SKIP_ASSET_CHECK=1
# to plan a DAG on a machine that does not hold the assets (e.g. inspecting a
# --dag off-cluster).
if not _os.environ.get("PROTFORGE_SKIP_ASSET_CHECK"):
    _asset_errors = _check_stage_assets(config)
    if _asset_errors:
        raise WorkflowError(
            "Install is incomplete — these are needed by the stages you enabled:"
            "\n  - " + "\n  - ".join(_asset_errors)
        )

# Pipeline start time (wall-clock)
_PIPELINE_START = _time.time()

RUN_MSA     = config["pipeline"].get("msa", True)
RUN_BOLTZ   = config["pipeline"].get("boltz", True)
# ESM-C embeddings + ESMFold2 structure prediction. Default OFF so a minimal
# MSA->Boltz config does not pull them in. Both run off the MSA-stage YAMLs.
RUN_ESMC    = config["pipeline"].get("esmc", False)
RUN_ESMFOLD = config["pipeline"].get("esmfold", False)
# ESMC SAE extraction. Gated by esmc.sae.enabled (independent of RUN_ESMC) so it
# can run standalone against YAMLs from a previous embedding run.
RUN_ESMC_SAE = config.get("esmc", {}).get("sae", {}).get("enabled", False)
# OpenFold3 structure prediction. Runs off the MSA-stage YAMLs in parallel with
# Boltz; default OFF. Outputs -> sequences/{seq}/openfold/.
RUN_OPENFOLD = config["pipeline"].get("openfold", False)
OUTPUT    = config["output"]["parent_dir"]
SLURM_CFG = config.get("slurm", {})
SEQUENCES_DIR = f"{OUTPUT}/sequences"

# SLURM email notifications (reads slurm.email from config)
SLURM_EMAIL = SLURM_CFG.get("email", "")

def slurm_extra(gpu=False, gpu_count=1):
    """Build slurm_extra string with optional GPU and mail flags.

    Each sbatch flag must be individually quoted so the shell
    passes them as separate arguments to sbatch. `gpu_count` sets how many
    GPUs the node request asks for (e.g. OpenFold's multi-GPU batched jobs).
    """
    parts = []
    if gpu:
        parts.append(f"'--gpus-per-node={int(gpu_count)}'")
    if SLURM_EMAIL:
        parts.append(f"'--mail-type=END,FAIL'")
        parts.append(f"'--mail-user={SLURM_EMAIL}'")
    return " ".join(parts) if parts else "''"


# Rule-side wrappers around workflow/scripts/snake_helpers.py — the rule files
# call stage_resource("boltz", ...) and we supply SLURM_CFG here.
def stage_resource(stage, key, default):
    return _stage_resource_impl(SLURM_CFG, stage, key, default)


def stage_uses_gpu(stage, default):
    return _stage_uses_gpu_impl(SLURM_CFG, stage, default)


def chunk_resource(chunks_tsv_path, chunk_id, key, default):
    """Per-chunk SLURM resource lookup from <stage>_chunks/chunks.tsv.

    Returns `default` when binning is disabled (chunks.tsv missing) or the
    row/key is absent. Used inside rule `resources:` callables."""
    return _chunk_resource_impl(chunks_tsv_path, chunk_id, key, default)


def binning_args(stage_cfg: dict, *, stage_name: str = "?") -> str:
    """Render `<stage>.binning` config block as a CLI flag string for the chunker.

    Returns "" when binning is disabled OR when enabled-but-incomplete (missing
    `bins:` recipe). The latter is treated as a soft fallback to non-binning
    mode with a stderr warning, so the workflow still loads. Hard error only
    on inconsistencies that look like real config bugs (mode=thresholds with
    a mismatched threshold/bin count, or per-bin entries missing required keys)."""
    binning = (stage_cfg or {}).get("binning") or {}
    if not binning.get("enabled", False):
        return ""
    bins = binning.get("bins") or []
    if not bins:
        _sys.stderr.write(
            f"WARN [{stage_name}]: binning.enabled=true but binning.bins is empty. "
            f"Falling back to non-binning chunking. Run the webapp estimator's "
            f"'Apply to session config' to populate the recipe, or unset "
            f"{stage_name}.binning.enabled.\n"
        )
        return ""
    mems = []
    runtimes = []
    for i, b in enumerate(bins):
        for k in ("mem_mb", "runtime_min"):
            if k not in b:
                raise ValueError(f"{stage_name}.binning.bins[{i}] missing key '{k}'")
        mems.append(int(b["mem_mb"]))
        runtimes.append(int(b["runtime_min"]))
    chunks_per_bin = int(binning.get("chunks_per_bin", 1))
    if chunks_per_bin < 1:
        raise ValueError(
            f"{stage_name}.binning.chunks_per_bin must be >= 1, got {chunks_per_bin}"
        )
    parts = [
        "--enable_binning",
        f"--bin_mode {binning.get('mode', 'quantile')}",
        f"--num_bins {int(binning.get('num_bins', len(bins)))}",
        f"--chunks_per_bin {chunks_per_bin}",
        f"--bin_mem_mb {','.join(str(x) for x in mems)}",
        f"--bin_runtime_min {','.join(str(x) for x in runtimes)}",
    ]
    if binning.get("mode") == "thresholds":
        thresholds = binning.get("thresholds") or []
        if len(thresholds) + 1 != len(bins):
            raise ValueError(
                f"{stage_name}.binning.mode=thresholds requires "
                f"len(thresholds)+1 == len(bins); got {len(thresholds)} "
                f"thresholds and {len(bins)} bins."
            )
        parts.append(f"--bin_thresholds {','.join(str(int(t)) for t in thresholds)}")
    return " ".join(parts)

# Container support (set .sif paths in config to enable)
CONTAINERS = config.get("containers", {})

# Container runtime: "singularity" | "apptainer" | "auto" (default).
# "auto" picks whichever binary is on PATH, preferring `singularity` (the
# Kempner-handbook-documented name; on most modern HPCs it's actually
# Apptainer's compat symlink, so the choice is cosmetic). Override via
# config: containers.runtime: apptainer
import shutil as _shutil
_RT = CONTAINERS.get("runtime", "auto")
_ALLOWED_RUNTIMES = ("auto", "singularity", "apptainer")
if _RT not in _ALLOWED_RUNTIMES:
    raise ValueError(
        f"containers.runtime must be one of {_ALLOWED_RUNTIMES}, got {_RT!r}. "
        f"This value is interpolated into shell commands — it is not free-form."
    )
if _RT == "auto":
    CONTAINER_RUNTIME = "singularity" if _shutil.which("singularity") else (
        "apptainer" if _shutil.which("apptainer") else "singularity"
    )
else:
    CONTAINER_RUNTIME = _RT

import shlex as _shlex


def container_sif(stage):
    """Resolve the SIF path for a stage: per-stage `containers.<stage>` wins,
    else the optional shared `containers.gpu`. Returns "" when none is set.

    Every GPU stage (MSA, Boltz, ESMC, ESMFold2) builds its own
    `<runtime> exec --nv ...` command with per-job bind mounts in its shell
    block, so this just hands back the image path to drop in.
    """
    return CONTAINERS.get(stage, "") or CONTAINERS.get("gpu", "")

if RUN_MSA:
    include: "workflow/rules/msa.smk"
if RUN_BOLTZ:
    include: "workflow/rules/boltz.smk"
if RUN_ESMC:
    include: "workflow/rules/esmc.smk"
if RUN_ESMC_SAE:
    include: "workflow/rules/esmc_sae.smk"
if RUN_ESMFOLD:
    include: "workflow/rules/esmfold.smk"
if RUN_OPENFOLD:
    include: "workflow/rules/openfold.smk"


def get_targets():
    """Build the list of final sentinel files based on pipeline toggles."""
    targets = []
    if RUN_MSA:
        targets.append(f"{OUTPUT}/.msa_complete")
    if RUN_BOLTZ:
        targets.append(f"{OUTPUT}/.boltz_complete")
    if RUN_ESMC:
        # One sentinel per configured ESMC model size.
        targets += [f"{OUTPUT}/.esmc_{size}_complete" for size in ESMC_SIZES]
    if RUN_ESMC_SAE:
        # One sentinel per ESMC model size we extract SAE activations for.
        targets += [f"{OUTPUT}/.esmc_sae_{size}_complete" for size in ESMC_SAE_SIZES]
    if RUN_ESMFOLD:
        targets.append(f"{OUTPUT}/.esmfold_complete")
    if RUN_OPENFOLD:
        targets.append(f"{OUTPUT}/.openfold_complete")
    return targets


rule all:
    input:
        get_targets(),


def _write_benchmark_summary(status):
    """Collect per-rule benchmarks and write a summary report."""
    from pathlib import Path
    elapsed = _time.time() - _PIPELINE_START
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    bench_dir = Path(OUTPUT) / "benchmarks"
    lines = [
        f"ProtForge Pipeline — {status}",
        "=" * 50,
        f"Total wall-clock time: {int(hours)}h {int(minutes)}m {seconds:.1f}s ({elapsed:.1f}s)",
        "",
    ]

    # Aggregate per-rule benchmark files
    if bench_dir.is_dir():
        import csv
        stage_times = {}
        for bench_file in sorted(bench_dir.rglob("*.tsv")):
            stage = bench_file.parent.name  # e.g. msa, boltz, esmc, esmfold
            try:
                with open(bench_file) as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        wall = float(row.get("s", 0))
                        stage_times.setdefault(stage, []).append(wall)
            except Exception:
                continue

        if stage_times:
            lines.append("Per-stage breakdown (sum of rule wall-clock times):")
            lines.append("-" * 50)
            lines.append(f"{'Stage':<15} {'Total (s)':>12} {'# Rules':>10} {'Avg (s)':>10}")
            total_rule_time = 0
            for stage in ["msa", "boltz", "esmc", "esmc_sae", "esmfold", "openfold"]:
                if stage not in stage_times:
                    continue
                times = stage_times[stage]
                total = sum(times)
                total_rule_time += total
                avg = total / len(times) if times else 0
                h, r = divmod(total, 3600)
                m, s = divmod(r, 60)
                lines.append(
                    f"{stage:<15} {total:>12.1f} {len(times):>10} {avg:>10.1f}"
                    f"   ({int(h)}h {int(m)}m {s:.0f}s)"
                )
            lines.append("-" * 50)
            h, r = divmod(total_rule_time, 3600)
            m, s = divmod(r, 60)
            lines.append(f"{'Sum rules':<15} {total_rule_time:>12.1f}{'':>10}{'':>10}   ({int(h)}h {int(m)}m {s:.0f}s)")
            lines.append("")

    report_path = Path(OUTPUT) / "benchmark_summary.txt"
    report_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nBenchmark summary written to: {report_path}")


onstart:
    # Provenance manifest: record git commit, resolved config, container
    # digests, and model SHAs so a run can be reproduced/audited later.
    try:
        from write_run_manifest import write_manifest as _write_manifest
        _mpath = _write_manifest(
            OUTPUT,
            config,
            repo_dir=workflow.basedir,
            runtime=CONTAINER_RUNTIME,
        )
        print(f"Wrote run provenance manifest: {_mpath}")
    except Exception as _e:  # never block a run on provenance
        _sys.stderr.write(f"WARN: could not write run_manifest.json: {_e}\n")


onsuccess:
    _write_benchmark_summary("COMPLETED SUCCESSFULLY")

onerror:
    _write_benchmark_summary("FAILED")
