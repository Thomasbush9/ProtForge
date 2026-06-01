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


def _parse_bind(entry):
    """Parse one bind_paths entry into a shell-safe -B flag.

    Accepted forms (longest-match first):
      - "host:container:ro"  (or :rw)        explicit indirection + mode
      - "host:container"                     explicit indirection, mode rw
      - "host:ro"            (or :rw)        host:host bind at given mode
      - "host"                               host:host bind, mode rw

    Disambiguation: when an entry has two parts, a second part equal to
    `ro` or `rw` is treated as the mode (host:host:mode); otherwise it's
    the container-side path. This lets the template default stay readable
    (`/n/.../db:ro`) without inventing a new separator.

    The full `-B` argument is `shlex.quote`d so paths with spaces or shell
    metachars can't break the rule's bash line. Empty entries are ignored.
    Unknown modes are rejected.
    """
    entry = entry.strip()
    if not entry:
        return None
    parts = entry.split(":")
    if len(parts) == 3:
        host, cont, mode = parts
        if mode not in ("ro", "rw"):
            raise ValueError(
                f"containers.bind_paths: invalid mode {mode!r} in {entry!r} "
                f"(expected 'ro' or 'rw')"
            )
        return f"-B {_shlex.quote(f'{host}:{cont}:{mode}')}"
    if len(parts) == 2:
        host, second = parts
        if second in ("ro", "rw"):
            return f"-B {_shlex.quote(f'{host}:{host}:{second}')}"
        return f"-B {_shlex.quote(f'{host}:{second}')}"
    return f"-B {_shlex.quote(entry)}"


def container_cmd(stage, extra_env=""):
    """Return '<runtime> exec --nv --cleanenv ... <extra_env> -B ... sif',
    or '' for legacy (non-container) mode.

    Args:
      stage: stage name used to resolve the SIF (containers.<stage>, else
        containers.gpu fallback).
      extra_env: optional `--env KEY=VALUE [--env ...]` string the caller
        wants injected *between* the runtime flags and the SIF path. The
        whole string lands BEFORE the SIF so Singularity treats it as a
        runtime option, not part of the in-container command. Values may
        contain `$VAR` references — those are resolved by bash at rule
        runtime, not at Snakemake plan time.

    Audit hardening (H1, H3, H5 — see vault container-audit.md):
      - --cleanenv: strip host env to prevent PYTHONPATH/CONDA_PREFIX leaks.
        Rules that need to forward a host env var into the container must
        pass it through `extra_env` so it lands before the SIF (Singularity
        rejects --env after the image).
      - bind_paths entries support 'host:container:ro' for read-only mounts
        (DBs should be :ro). See _parse_bind for accepted shorthand.
      - SLURM_TMPDIR (or /tmp if unset) is bound at /tmp and TMPDIR=/tmp is
        propagated, so tools that write large temp files (Boltz, Triton,
        HF) land on node-local scratch instead of the container's tmpfs.
        The ${{SLURM_TMPDIR:-/tmp}} expansion is shell-expanded at rule
        runtime (it appears inside the rule's bash shell block).
    """
    # Resolve SIF: per-stage image (containers.<stage>, e.g. containers.boltz)
    # is the primary path. `containers.gpu` remains as an optional shared
    # fallback for users who bundle several stages into one image.
    sif = CONTAINERS.get(stage, "") or CONTAINERS.get("gpu", "")
    if not sif:
        return ""
    binds = [_parse_bind(p) for p in BIND_PATHS.split(",")]
    binds = [b for b in binds if b is not None]
    # Node-local scratch for tmp (shell-expanded by the rule's bash).
    binds.append('-B "${SLURM_TMPDIR:-/tmp}":/tmp')
    flags = [
        CONTAINER_RUNTIME, "exec",
        "--nv",
        "--cleanenv",
        "--env", "TMPDIR=/tmp",
    ]
    parts = flags + binds
    if extra_env:
        # extra_env is a pre-rendered flag string (e.g. "--env FOO=bar
        # --env BAZ=$BAZ"). Trust the caller to have quoted any path
        # components — we just inject it verbatim before the SIF.
        parts.append(extra_env)
    # `sif` lands at the end of a rule's bash command line, so quote it
    # in case the user's path contains spaces or shell metachars.
    parts.append(_shlex.quote(sif))
    return " ".join(parts)

if RUN_MSA:
    include: "workflow/rules/msa.smk"
if RUN_BOLTZ:
    include: "workflow/rules/boltz.smk"
if RUN_ESM:
    include: "workflow/rules/esm.smk"
if RUN_ESMFOLD:
    include: "workflow/rules/esmfold.smk"


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
            for stage in ["msa", "boltz", "esm", "esmfold"]:
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
