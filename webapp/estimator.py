"""
Resource estimator for the ProtForge pipeline.

Given input statistics from `webapp/validate.py` (count, sequence lengths)
and a Snakemake config, produce per-stage SLURM resource recommendations
(mem_mb, runtime_min, cpus, gpus, partition, chunk size) that can be
written back into the session config with `apply_estimate_to_config`.

Pure functions only — no Streamlit deps. Importable from notebooks and
covered by webapp/tests/test_estimator.py.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

import yaml


HERE = Path(__file__).parent
DEFAULT_SCALING_PATH = HERE / "scaling_models.yaml"
CALIBRATED_SCALING_PATH = HERE / "scaling_models.calibrated.yaml"

GPU_USING_STAGES = {"msa", "boltz", "esm", "esmfold"}
ALL_STAGES = ["msa", "boltz", "esm", "esmfold", "es"]


# --- Data containers -------------------------------------------------------


@dataclass
class InputStats:
    count: int
    min_len: int
    max_len: int
    mean_len: float
    p95_len: int
    total_residues: int
    file_type: str  # "fasta" | "yaml"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class StageEstimate:
    stage: str
    mem_mb: int
    runtime_min: int
    cpus: int
    gpus: int
    gpu_type: str | None
    partition: str
    chunk_size: int
    num_chunks: int
    total_node_hours: float
    calibrated: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# --- Loading scaling models -----------------------------------------------


def load_scaling_models(path: Path | None = None) -> dict:
    """Load scaling_models.yaml. If a calibrated file exists alongside it,
    its values shadow the defaults stage-by-stage, GPU-by-GPU.
    """
    base_path = path or DEFAULT_SCALING_PATH
    with open(base_path) as f:
        models = yaml.safe_load(f)

    if path is None and CALIBRATED_SCALING_PATH.exists():
        with open(CALIBRATED_SCALING_PATH) as f:
            calibrated = yaml.safe_load(f) or {}
        _merge_calibration(models, calibrated)

    return models


def _merge_calibration(base: dict, calibrated: dict) -> None:
    """In-place merge: per-GPU coefficient blocks from calibrated override base."""
    for stage, cal_stage in calibrated.items():
        if stage not in base or not isinstance(cal_stage, dict):
            continue
        cal_per_gpu = cal_stage.get("per_gpu", {})
        base_per_gpu = base[stage].setdefault("per_gpu", {})
        for gpu, coeffs in cal_per_gpu.items():
            base_per_gpu[gpu] = coeffs


# --- Stats from validation results ----------------------------------------


def compute_input_stats(
    fasta_results: Iterable[dict] | None = None,
    yaml_results: Iterable[dict] | None = None,
) -> InputStats:
    """Compute summary stats from validate.py output.

    Either pass `fasta_results` (each dict has 'total_residues' from
    validate_fasta) or `yaml_results` (each has 'sequence_length' from
    validate_yaml). Returns InputStats describing the valid inputs only.
    """
    fasta_results = list(fasta_results or [])
    yaml_results = list(yaml_results or [])

    if fasta_results and yaml_results:
        raise ValueError("Pass either fasta_results or yaml_results, not both")

    if fasta_results:
        lengths = [r["total_residues"] for r in fasta_results if r.get("valid")]
        file_type = "fasta"
    else:
        lengths = [r["sequence_length"] for r in yaml_results if r.get("valid")]
        file_type = "yaml"

    if not lengths:
        return InputStats(0, 0, 0, 0.0, 0, 0, file_type)

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    p95_idx = max(0, math.ceil(0.95 * n) - 1)

    return InputStats(
        count=n,
        min_len=lengths_sorted[0],
        max_len=lengths_sorted[-1],
        mean_len=sum(lengths_sorted) / n,
        p95_len=lengths_sorted[p95_idx],
        total_residues=sum(lengths_sorted),
        file_type=file_type,
    )


# --- Per-stage formulas ---------------------------------------------------


def _eval_mem(coeffs: dict, *, p95_len: int, mean_len: float, total_residues: int,
              count: int) -> float:
    """Evaluate the memory polynomial. Coefficient names are documented in
    scaling_models.yaml; missing terms default to 0.
    """
    base = coeffs.get("base", 0)
    per_residue = coeffs.get("per_residue", 0)
    alpha_L2 = coeffs.get("alpha_L2", 0)
    per_struct = coeffs.get("per_struct", 0)
    return (
        base
        + per_residue * p95_len
        + alpha_L2 * (p95_len ** 2)
        + per_struct * count
    )


def _eval_time_per_seq(coeffs: dict, *, mean_len: float) -> float:
    """Time per sequence in seconds, evaluated at mean length."""
    base = coeffs.get("base", 0)
    alpha_L = coeffs.get("alpha_L", 0)
    alpha_L2 = coeffs.get("alpha_L2", 0)
    return base + alpha_L * mean_len + alpha_L2 * (mean_len ** 2)


def _eval_time_per_chunk(coeffs: dict, *, mean_len: float, chunk_size: int,
                         total_residues_per_chunk: float) -> float:
    """Time for a single MSA-style chunk (per_seq + per_residue terms)."""
    base = coeffs.get("base", 0)
    per_seq = coeffs.get("per_seq", 0)
    per_residue = coeffs.get("per_residue", 0)
    return base + per_seq * chunk_size + per_residue * total_residues_per_chunk


# --- Main estimator -------------------------------------------------------


def _pick_gpu_type(stage_models: dict, requested: str | None,
                  estimated_mem_gb: float, gpu_specs: dict) -> tuple[str, list[str]]:
    """Choose which GPU type to size against.

    If `requested` is given (and exists in per_gpu), use it.
    Otherwise: walk `tier` from low to high and pick the first whose
    mem_gb covers the estimate.
    """
    notes: list[str] = []
    available = list(stage_models.get("per_gpu", {}).keys())
    if not available:
        return "cpu", notes

    if requested:
        spec = gpu_specs.get(requested, {})
        if spec and spec.get("mem_gb", 0) < estimated_mem_gb:
            notes.append(
                f"Requested {requested} ({spec['mem_gb']}GB) is below estimated "
                f"need ({estimated_mem_gb:.1f}GB) — risk of OOM."
            )
        if requested not in available:
            notes.append(
                f"No calibrated coefficients for {requested}; partition will use "
                f"{requested} but timing/mem estimate uses a proxy GPU."
            )
        return requested, notes

    # Auto pick: ascend tiers
    candidates = sorted(
        [(g, gpu_specs.get(g, {}).get("tier", 99), gpu_specs.get(g, {}).get("mem_gb", 0))
         for g in available if g in gpu_specs],
        key=lambda x: x[1],
    )
    for gpu, _tier, mem_gb in candidates:
        if mem_gb >= estimated_mem_gb:
            return gpu, notes

    # Nothing fits — return the largest we know
    if candidates:
        gpu, _, mem_gb = candidates[-1]
        notes.append(
            f"Estimated mem {estimated_mem_gb:.1f}GB exceeds largest known GPU "
            f"({gpu}: {mem_gb}GB) — job may OOM."
        )
        return gpu, notes

    # Fall back to default
    default = stage_models.get("default_gpu") or available[0]
    return default, notes


def estimate_stage(
    stage: str,
    stats: InputStats,
    config: dict,
    scaling: dict,
    gpu_preference: str | None = None,
) -> StageEstimate:
    """Estimate SLURM resources + chunk size for one pipeline stage.

    `gpu_preference` is "auto" or None for auto-pick, or a key from gpu_specs
    (e.g. "h100"). `config` is the Snakemake config dict (for boltz multipliers,
    es config, etc.).
    """
    if stats.count == 0:
        return StageEstimate(
            stage=stage, mem_mb=0, runtime_min=0, cpus=0, gpus=0,
            gpu_type=None, partition="", chunk_size=0, num_chunks=0,
            total_node_hours=0.0,
            notes=["No valid inputs — skipping estimate"],
        )

    stage_models = scaling.get(stage)
    if stage_models is None:
        raise KeyError(f"No scaling model for stage '{stage}'")

    margins = scaling.get("margins", {})
    mem_safety = margins.get("mem_safety", 1.3)
    time_safety = margins.get("time_safety", 1.5)
    min_runtime_min = margins.get("min_runtime_min", 15)

    gpu_specs = scaling.get("gpu_specs", {})
    requested = None if (gpu_preference in (None, "auto", "")) else gpu_preference

    # Pick GPU based on a *first-pass* mem estimate using the largest GPU's
    # coefficients (so we don't loop). For ES (cpu-only), skip this and use 'cpu'.
    if stage == "es":
        gpu_type = "cpu"
        gpu_notes: list[str] = []
        coeffs_per_gpu = stage_models["per_gpu"]["cpu"]
    else:
        # Reasonable "any GPU" coefficients — use h100 if present, else default_gpu
        per_gpu = stage_models["per_gpu"]
        seed_gpu = "h100" if "h100" in per_gpu else stage_models.get("default_gpu") or next(iter(per_gpu))
        seed_mem = _eval_mem(
            per_gpu[seed_gpu]["mem_mb"],
            p95_len=stats.p95_len, mean_len=stats.mean_len,
            total_residues=stats.total_residues, count=stats.count,
        )
        seed_mem_gb = (seed_mem * mem_safety) / 1024
        gpu_type, gpu_notes = _pick_gpu_type(stage_models, requested, seed_mem_gb, gpu_specs)
        coeffs_per_gpu = per_gpu.get(gpu_type)
        if coeffs_per_gpu is None:
            # Fall back to default_gpu's coefficients but keep the chosen partition
            fallback_gpu = stage_models.get("default_gpu") or seed_gpu
            coeffs_per_gpu = per_gpu[fallback_gpu]
            gpu_notes.append(
                f"No per-GPU coefficients for {gpu_type}; using {fallback_gpu} as proxy."
            )

    # Memory
    mem_raw = _eval_mem(
        coeffs_per_gpu["mem_mb"],
        p95_len=stats.p95_len, mean_len=stats.mean_len,
        total_residues=stats.total_residues, count=stats.count,
    )
    mem_mb = int(math.ceil(mem_raw * mem_safety))

    # Per-stage time + chunk size logic
    target_chunk_min = stage_models.get("target_chunk_runtime_min", 30)
    max_chunk = stage_models.get("max_chunk_size", stats.count)
    min_chunk = stage_models.get("min_chunk_size", 1)

    if stage == "boltz":
        boltz_cfg = config.get("boltz", {})
        recycling = boltz_cfg.get("recycling_steps", 10)
        samples = boltz_cfg.get("diffusion_samples", 25)
        num_runs = boltz_cfg.get("num_runs", 1)
        # YAML coefficients are calibrated at recycling=10, samples=25 (250 work
        # units). Scale linearly with the user's settings. num_runs creates
        # separate SLURM jobs, so it multiplies num_chunks below, not per-seq time.
        per_seq_sec = _eval_time_per_seq(
            coeffs_per_gpu["runtime_sec_per_seq"], mean_len=stats.mean_len
        ) * (recycling * samples) / 250
        per_seq_sec = max(per_seq_sec, 1.0)
        chunk_size = max(min_chunk, min(max_chunk,
            int(math.floor((target_chunk_min * 60) / per_seq_sec))))
        chunk_size = max(chunk_size, 1)
        runtime_sec = per_seq_sec * chunk_size * time_safety
        num_chunks = math.ceil(stats.count / chunk_size) * num_runs
    elif stage in ("esm", "esmfold"):
        per_seq_sec = _eval_time_per_seq(
            coeffs_per_gpu["runtime_sec_per_seq"], mean_len=stats.mean_len
        )
        per_seq_sec = max(per_seq_sec, 0.5)
        chunk_size = max(min_chunk, min(max_chunk,
            int(math.floor((target_chunk_min * 60) / per_seq_sec))))
        runtime_sec = per_seq_sec * chunk_size * time_safety
        num_chunks = math.ceil(stats.count / chunk_size)
    elif stage == "msa":
        # MSA chunks: time = base + per_seq*N + per_residue*sum(L_in_chunk)
        # Iteratively pick chunk size such that chunk-runtime ~ target.
        # Use mean_len * chunk_size as residue total.
        coeffs_t = coeffs_per_gpu["runtime_sec_per_chunk"]
        # Solve: target = base + per_seq*k + per_residue*mean*k -> k
        per_seq = coeffs_t.get("per_seq", 0) + coeffs_t.get("per_residue", 0) * stats.mean_len
        base = coeffs_t.get("base", 0)
        target_sec = target_chunk_min * 60
        if per_seq > 0:
            chunk_size = int(math.floor((target_sec - base) / per_seq))
        else:
            chunk_size = max_chunk
        chunk_size = max(min_chunk, min(max_chunk, max(chunk_size, 1)))
        runtime_sec = (
            _eval_time_per_chunk(
                coeffs_t,
                mean_len=stats.mean_len,
                chunk_size=chunk_size,
                total_residues_per_chunk=stats.mean_len * chunk_size,
            )
            * time_safety
        )
        num_chunks = math.ceil(stats.count / chunk_size)
    elif stage == "es":
        coeffs_t = coeffs_per_gpu["runtime_sec"]
        runtime_sec = (
            coeffs_t.get("base", 0)
            + coeffs_t.get("per_struct", 0) * stats.count
            + coeffs_t.get("per_residue", 0) * stats.total_residues
        ) * time_safety
        chunk_size = stats.count  # monolithic
        num_chunks = 1
    else:
        raise ValueError(f"Unknown stage: {stage}")

    runtime_min = max(min_runtime_min, int(math.ceil(runtime_sec / 60)))

    # Partition: GPU type's partition unless config has explicit override
    partition = ""
    config_slurm = config.get("slurm", {})
    stage_override = config_slurm.get(stage, {}).get("partition", "")
    if stage_override:
        partition = stage_override
    elif gpu_type and gpu_type != "cpu":
        partition = gpu_specs.get(gpu_type, {}).get("partition", "")
    if not partition:
        partition = config_slurm.get("partition", "")

    # Total cost (node-hours): one node per SLURM job × runtime
    total_node_hours = (runtime_min * num_chunks) / 60

    return StageEstimate(
        stage=stage,
        mem_mb=mem_mb,
        runtime_min=runtime_min,
        cpus=stage_models.get("cpus", 4),
        gpus=stage_models.get("gpus", 0),
        gpu_type=None if gpu_type == "cpu" else gpu_type,
        partition=partition,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        total_node_hours=round(total_node_hours, 2),
        calibrated=False,  # filled in by caller if scaling came from .calibrated.yaml
        notes=gpu_notes,
    )


def estimate_all_stages(
    stats: InputStats,
    config: dict,
    scaling: dict | None = None,
    gpu_preferences: dict | None = None,
) -> dict[str, StageEstimate]:
    """Estimate all enabled stages in one call.

    `gpu_preferences` maps stage name → "auto" | "a100" | "h100" | "h200".
    Reads the `pipeline.<stage>` toggles from `config` to decide which to skip.
    """
    if scaling is None:
        scaling = load_scaling_models()

    pipeline = config.get("pipeline", {})
    gpu_preferences = gpu_preferences or {}

    out: dict[str, StageEstimate] = {}
    for stage in ALL_STAGES:
        if not pipeline.get(stage, False):
            continue
        # If running ESMFold on raw fasta and stats are YAML (or vice versa), still estimate
        out[stage] = estimate_stage(
            stage, stats, config, scaling, gpu_preferences.get(stage)
        )
    return out


# --- Partition picker (exposed for tests + UI) ----------------------------


def pick_partition(
    stage: str,
    stats: InputStats,
    config: dict,
    scaling: dict | None = None,
    gpu_preference: str | None = None,
) -> tuple[str, str | None]:
    """Standalone partition picker: returns (partition, gpu_type)."""
    est = estimate_stage(
        stage, stats, config, scaling or load_scaling_models(), gpu_preference
    )
    return est.partition, est.gpu_type


# --- Apply estimate to session config -------------------------------------


def apply_estimate_to_config(
    config_path: Path,
    estimates: dict[str, StageEstimate],
    backup: bool = True,
) -> Path:
    """Write per-stage resources + chunk sizes into the session config YAML.

    Uses two namespaces:
      slurm.resources.<stage>.{mem_mb, runtime, cpus_per_task, gpus}
      slurm.<stage>.partition
      <stage>.max_files_per_job  (msa, boltz)
      <stage>.num_chunks         (esm, esmfold)
    Preserves all other keys. Returns the backup path (or original path).
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    slurm = cfg.setdefault("slurm", {})
    slurm_resources = slurm.setdefault("resources", {})

    chunk_key = {
        "msa": "max_files_per_job",
        "boltz": "max_files_per_job",
        "esm": "num_chunks",
        "esmfold": "num_chunks",
        "es": None,
    }

    for stage, est in estimates.items():
        if est.num_chunks == 0:
            continue
        # SLURM resources
        slurm_resources[stage] = {
            "mem_mb": est.mem_mb,
            "runtime": est.runtime_min,
            "cpus_per_task": est.cpus,
            "gpus": est.gpus,
        }
        # Stage partition (only set if estimator picked one)
        if est.partition:
            slurm.setdefault(stage, {})["partition"] = est.partition

        # Chunk size into the stage's own config block
        ck = chunk_key.get(stage)
        if ck:
            stage_cfg = cfg.setdefault(stage, {})
            if stage in ("msa", "boltz"):
                stage_cfg[ck] = est.chunk_size
            else:  # esm, esmfold — num_chunks is the *number of jobs*
                stage_cfg[ck] = est.num_chunks

    backup_path = config_path
    if backup:
        backup_dir = config_path.parent / ".config_backups"
        backup_dir.mkdir(exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"config_{ts}.yaml"
        shutil.copy2(config_path, backup_path)

    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    return backup_path


# --- Calibration ----------------------------------------------------------


def recalibrate_from_benchmarks(
    benchmarks_dir: Path,
    chunk_stats_dir: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    """Fit per-stage scaling coefficients from Snakemake benchmark TSVs.

    Expects:
      - `benchmarks_dir`/<stage>/*.tsv with Snakemake's columns ('s', 'max_rss', ...)
      - Optional `chunk_stats_dir`/<stage>/chunk_<id>_stats.json with
        {'mean_len', 'p95_len', 'count', 'total_residues', 'gpu_type'}.
    If chunk stats are missing we fall back to using the global input stats
    that were used at run time (must be passed in or skipped).

    Writes refined per-GPU coefficients to scaling_models.calibrated.yaml.
    Currently fits only `runtime_sec_per_seq.base` and `mem_mb.base` per GPU
    type as a v1 sanity check; richer regression is a follow-up.
    """
    import csv
    benchmarks_dir = Path(benchmarks_dir)
    output_path = output_path or CALIBRATED_SCALING_PATH

    fitted: dict = {}

    for stage_dir in benchmarks_dir.iterdir():
        if not stage_dir.is_dir():
            continue
        stage = stage_dir.name
        if stage not in ALL_STAGES:
            continue

        runtimes: list[float] = []
        max_rss_kb: list[float] = []
        for tsv in stage_dir.glob("*.tsv"):
            try:
                with open(tsv) as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        runtimes.append(float(row.get("s", 0)))
                        # Snakemake writes max_rss in MB (column 'max_rss')
                        rss = row.get("max_rss") or row.get("max_uss") or 0
                        max_rss_kb.append(float(rss) * 1024 if rss else 0)
            except Exception:
                continue

        if not runtimes:
            continue

        # v1 calibration: just store mean runtime + mean RSS as a "base" fit.
        # Later: regress against chunk_stats.json to fit alpha_L2 etc.
        avg_runtime = sum(runtimes) / len(runtimes)
        avg_mem_mb = (sum(max_rss_kb) / len(max_rss_kb) / 1024) if max_rss_kb else 0
        # Without chunk_stats, we don't know the GPU type — store under "_observed"
        fitted[stage] = {
            "per_gpu": {
                "_observed": {
                    "runtime_sec_per_seq": {"base": round(avg_runtime, 2)},
                    "mem_mb": {"base": round(avg_mem_mb, 0)},
                }
            },
            "_meta": {
                "n_jobs": len(runtimes),
                "source": str(benchmarks_dir),
            }
        }

    if fitted:
        with open(output_path, "w") as f:
            yaml.safe_dump(fitted, f, sort_keys=False, default_flow_style=False)

    return fitted
