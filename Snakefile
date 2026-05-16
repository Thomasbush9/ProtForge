"""
ProtForge Snakemake Workflow
============================
Orchestrates the 4-stage protein prediction pipeline:
  MSA -> Boltz -> ESM -> ES

Usage:
  snakemake --profile profiles/slurm/           # Full pipeline via SLURM
  snakemake --profile profiles/slurm/ -n        # Dry run
  snakemake --profile profiles/slurm/ --dag | dot -Tpng > dag.png
  snakemake --profile profiles/slurm/ --rerun-incomplete
"""

import sys as _sys
import time as _time
from pathlib import Path as _Path

# Make workflow.scripts importable for any rule helpers we factor out.
_sys.path.insert(0, str(_Path(workflow.basedir) / "workflow" / "scripts"))
from snake_helpers import stage_resource as _stage_resource_impl
from snake_helpers import stage_uses_gpu as _stage_uses_gpu_impl
from binning import chunk_resource as _chunk_resource_impl

configfile: "config.yaml"

# Pipeline start time (wall-clock)
_PIPELINE_START = _time.time()

RUN_MSA     = config["pipeline"].get("msa", True)
RUN_BOLTZ   = config["pipeline"].get("boltz", True)
RUN_ESM     = config["pipeline"].get("esm", True)
RUN_ESMFOLD = config["pipeline"].get("esmfold", False)
RUN_ES      = config["pipeline"].get("es", True)
OUTPUT    = config["output"]["parent_dir"]
SLURM_CFG = config.get("slurm", {})
SEQUENCES_DIR = f"{OUTPUT}/sequences"

# SLURM email notifications (reads slurm.email from config)
SLURM_EMAIL = SLURM_CFG.get("email", "")

def slurm_extra(gpu=False):
    """Build slurm_extra string with optional GPU and mail flags.

    Each sbatch flag must be individually quoted so the shell
    passes them as separate arguments to sbatch.
    """
    parts = []
    if gpu:
        parts.append("'--gpus-per-node=1'")
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
BIND_PATHS = CONTAINERS.get("bind_paths", "/n/holylfs06,/n/home06")

# Container runtime: "singularity" | "apptainer" | "auto" (default).
# "auto" picks whichever binary is on PATH, preferring `singularity` (the
# Kempner-handbook-documented name; on most modern HPCs it's actually
# Apptainer's compat symlink, so the choice is cosmetic). Override via
# config: containers.runtime: apptainer
import shutil as _shutil
_RT = CONTAINERS.get("runtime", "auto")
if _RT == "auto":
    CONTAINER_RUNTIME = "singularity" if _shutil.which("singularity") else (
        "apptainer" if _shutil.which("apptainer") else "singularity"
    )
else:
    CONTAINER_RUNTIME = _RT

def container_cmd(stage):
    """Return '<runtime> exec --nv -B ... sif' prefix, or '' for legacy mode."""
    sif = CONTAINERS.get(stage, "")
    if sif:
        binds = " ".join(f"-B {p}" for p in BIND_PATHS.split(","))
        return f"{CONTAINER_RUNTIME} exec --nv {binds} {sif}"
    return ""

if RUN_MSA:
    include: "workflow/rules/msa.smk"
if RUN_BOLTZ:
    include: "workflow/rules/boltz.smk"
if RUN_ESM:
    include: "workflow/rules/esm.smk"
if RUN_ESMFOLD:
    include: "workflow/rules/esmfold.smk"
if RUN_ES:
    include: "workflow/rules/es.smk"


def get_targets():
    """Build the list of final sentinel files based on pipeline toggles."""
    targets = []
    if RUN_MSA:
        targets.append(f"{OUTPUT}/.msa_complete")
    if RUN_BOLTZ:
        targets.append(f"{OUTPUT}/.boltz_complete")
    if RUN_ESM:
        targets.append(f"{OUTPUT}/.esm_complete")
    if RUN_ESMFOLD:
        targets.append(f"{OUTPUT}/.esmfold_complete")
    if RUN_ES:
        targets.append(f"{OUTPUT}/es/.done")
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
            stage = bench_file.parent.name  # e.g. msa, boltz, esm, es
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
            for stage in ["msa", "boltz", "esm", "esmfold", "es"]:
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


onsuccess:
    _write_benchmark_summary("COMPLETED SUCCESSFULLY")

onerror:
    _write_benchmark_summary("FAILED")
