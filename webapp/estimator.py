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

GPU_USING_STAGES = {"msa", "boltz", "esmc", "esmfold"}
ALL_STAGES = ["msa", "boltz", "esmc", "esmfold"]


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
    # Full sorted lengths — kept so binning can compute true quantile cuts.
    # Optional / empty list when not populated (older callers).
    lengths_sorted: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("lengths_sorted", None)  # keep dict serialization small for UI
        return d


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
    # Optional bin plan when the user enables `<stage>.binning.enabled` — the
    # estimator computes per-bin chunk_size/mem/runtime sized for each bin's
    # representative length. None when binning is off.
    bin_plan: "BinPlan | None" = None

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class BinSpec:
    """Per-bin chunking + resource recommendation.

    Resources (mem_mb, runtime_min) are sized for the bin's worst-case length;
    `chunk_size` is derived from `chunks_per_bin` (a top-level BinPlan field)
    and the bin's actual `num_seqs`."""
    bin_idx: int
    num_seqs: int
    len_lo: int            # min observed length in bin (inclusive)
    len_hi: int            # max observed length in bin (inclusive)
    repr_len: int          # length used for sizing (typically bin's p95)
    chunk_size: int        # ceil(num_seqs / min(chunks_per_bin, num_seqs))
    mem_mb: int
    runtime_min: int

    @property
    def num_chunks(self) -> int:
        if self.num_seqs <= 0:
            return 0
        return math.ceil(self.num_seqs / max(1, self.chunk_size))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BinPlan:
    mode: str                          # "quantile" | "thresholds"
    num_bins: int
    chunks_per_bin: int                # how many chunks each non-empty bin is split into
    thresholds: list[int]              # effective length cuts (len = num_bins - 1)
    bins: list[BinSpec]

    @property
    def total_chunks(self) -> int:
        return sum(b.num_chunks for b in self.bins)

    @property
    def total_runtime_min(self) -> int:
        return sum(b.num_chunks * b.runtime_min for b in self.bins)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "num_bins": self.num_bins,
            "chunks_per_bin": self.chunks_per_bin,
            "thresholds": list(self.thresholds),
            "bins": [b.as_dict() for b in self.bins],
            "total_chunks": self.total_chunks,
            "total_runtime_min": self.total_runtime_min,
        }


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
        lengths_sorted=lengths_sorted,
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
                  stats: InputStats, mem_safety: float,
                  gpu_specs: dict) -> tuple[str, list[str]]:
    """Choose which GPU type to size against.

    If `requested` is given, use it (with an OOM-risk note if its own
    coefficients project past its mem_gb). Otherwise walk `tier` from low to
    high and pick the first candidate whose OWN coefficients fit inside its
    own mem_gb. Evaluating per-candidate avoids the bug where a smaller-GPU
    pick is justified by a more-efficient bigger-GPU's projection.
    """
    notes: list[str] = []
    per_gpu = stage_models.get("per_gpu", {})
    available = list(per_gpu.keys())
    if not available:
        return "cpu", notes

    def _own_mem_gb(gpu: str) -> float:
        """Mem the *gpu's own* coefficients project, with safety margin, in GB."""
        coeffs = per_gpu.get(gpu)
        if coeffs is None:
            return 0.0
        raw_mb = _eval_mem(
            coeffs["mem_mb"], p95_len=stats.p95_len, mean_len=stats.mean_len,
            total_residues=stats.total_residues, count=stats.count,
        )
        return raw_mb * mem_safety / 1024

    if requested:
        spec = gpu_specs.get(requested, {})
        cap = spec.get("mem_gb", 0)
        # If we have coefficients for the pinned GPU, check its own projection
        # against its own cap. Otherwise we can't tell — emit the no-coeffs note.
        if requested in per_gpu:
            req_mem_gb = _own_mem_gb(requested)
            if cap and req_mem_gb > cap:
                notes.append(
                    f"Pinned {requested} ({cap}GB) is below its own projected "
                    f"need ({req_mem_gb:.1f}GB) — risk of OOM."
                )
        else:
            notes.append(
                f"No calibrated coefficients for {requested}; partition will use "
                f"{requested} but timing/mem estimate uses a proxy GPU."
            )
        return requested, notes

    # Auto pick: ascend tiers, evaluating each candidate's OWN coefficients.
    candidates = sorted(
        [(g, gpu_specs.get(g, {}).get("tier", 99), gpu_specs.get(g, {}).get("mem_gb", 0))
         for g in available if g in gpu_specs],
        key=lambda x: x[1],
    )
    for gpu, _tier, cap in candidates:
        if cap >= _own_mem_gb(gpu):
            return gpu, notes

    # Nothing fits — return the largest we know
    if candidates:
        gpu, _, cap = candidates[-1]
        own = _own_mem_gb(gpu)
        notes.append(
            f"Estimated mem {own:.1f}GB exceeds largest known GPU "
            f"({gpu}: {cap}GB) — job may OOM."
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
    # coefficients (so we don't loop).
    if stage not in GPU_USING_STAGES:
        gpu_type = "cpu"
        gpu_notes: list[str] = []
        coeffs_per_gpu = stage_models["per_gpu"]["cpu"]
    else:
        per_gpu = stage_models["per_gpu"]
        gpu_type, gpu_notes = _pick_gpu_type(
            stage_models, requested, stats, mem_safety, gpu_specs)
        coeffs_per_gpu = per_gpu.get(gpu_type)
        if coeffs_per_gpu is None:
            # Fall back to default_gpu's coefficients but keep the chosen partition
            fallback_gpu = stage_models.get("default_gpu") or next(iter(per_gpu))
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
    elif stage in ("esmc", "esmfold"):
        # ESMC embeddings / ESMFold2: roughly per-sequence cost, chunked by
        # max_files_per_job. (ESMFold2 has no trunk-chunking, unlike the old v1
        # path, but the scaling block keeps a `_chunked` fallback for long L.)
        rt_block = coeffs_per_gpu["runtime_sec_per_seq"]
        if stage == "esmfold":
            chunk_threshold = stage_models.get("chunk_threshold", 10**9)
            chunked_block = coeffs_per_gpu.get("runtime_sec_per_seq_chunked")
            if chunked_block is not None and stats.p95_len >= chunk_threshold:
                rt_block = chunked_block
                gpu_notes.append(
                    f"ESMFold2: p95_len {stats.p95_len} ≥ {chunk_threshold} — "
                    f"using the slower long-sequence coefficients for the estimate."
                )
        per_seq_sec = _eval_time_per_seq(rt_block, mean_len=stats.mean_len)
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

    # Optional bin plan: if `<stage>.binning.enabled` is set in config, build a
    # per-bin chunking + resource plan now. The estimate's flat mem_mb/runtime
    # values stay as the safety-net fallback used when chunks.tsv is missing.
    stage_cfg = config.get(stage, {}) or {}
    binning_cfg = stage_cfg.get("binning") or {}
    bin_plan: BinPlan | None = None
    if binning_cfg.get("enabled", False) and stats.lengths_sorted:
        try:
            bp_kwargs = dict(
                stage=stage,
                stats=stats,
                coeffs_per_gpu=coeffs_per_gpu,
                stage_models=stage_models,
                mode=binning_cfg.get("mode", "quantile"),
                num_bins=int(binning_cfg.get("num_bins", 6)),
                chunks_per_bin=int(binning_cfg.get("chunks_per_bin", 1)),
                thresholds=binning_cfg.get("thresholds"),
                target_chunk_runtime_min=binning_cfg.get("target_chunk_runtime_min"),
                mem_safety=mem_safety,
                time_safety=time_safety,
                startup_overhead_min=int(binning_cfg.get("startup_overhead_min", 5)),
            )
            if stage == "boltz":
                boltz_cfg = config.get("boltz", {})
                bp_kwargs["boltz_recycling"] = boltz_cfg.get("recycling_steps", 10)
                bp_kwargs["boltz_samples"] = boltz_cfg.get("diffusion_samples", 25)
            bin_plan = compute_bin_plan(**bp_kwargs)
        except (ValueError, KeyError) as e:
            gpu_notes.append(f"Bin plan disabled: {e}")

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
        bin_plan=bin_plan,
    )


def _bin_lengths(lengths_sorted: list[int], thresholds: list[int]) -> list[list[int]]:
    """Partition a sorted length list into len(thresholds)+1 buckets."""
    n_bins = len(thresholds) + 1
    buckets: list[list[int]] = [[] for _ in range(n_bins)]
    for L in lengths_sorted:
        idx = 0
        for t in thresholds:
            if L < t:
                break
            idx += 1
        buckets[idx].append(L)
    return buckets


def _quantile_thresholds(lengths_sorted: list[int], num_bins: int) -> list[int]:
    """Mirror of binning.quantile_thresholds — kept here to avoid a cross-package
    import (binning lives in workflow/scripts/, not on the webapp's path).

    For num_bins=6 uses an upper-tail-weighted cut schedule
    (q25, q50, q75, q90, q95) — same shape as scripts/calibrate/subsample.py.
    Other num_bins use evenly-spaced quantiles."""
    if not lengths_sorted or num_bins < 2:
        return []
    n = len(lengths_sorted)
    if num_bins == 6:
        cuts = (0.25, 0.50, 0.75, 0.90, 0.95)  # 5 cuts → 6 bins
    else:
        cuts = tuple((i + 1) / num_bins for i in range(num_bins - 1))
    out = []
    for q in cuts:
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        out.append(lengths_sorted[idx])
    return out


def compute_bin_plan(
    *,
    stage: str,
    stats: InputStats,
    coeffs_per_gpu: dict,
    stage_models: dict,
    mode: str = "quantile",
    num_bins: int = 6,
    chunks_per_bin: int = 1,
    thresholds: list[int] | None = None,
    target_chunk_runtime_min: int | None = None,
    mem_safety: float = 1.3,
    time_safety: float = 1.3,
    startup_overhead_min: int = 5,
    boltz_recycling: int = 10,
    boltz_samples: int = 25,
) -> BinPlan:
    """Build a per-bin chunking + resource plan for one stage.

    Per-bin chunk_size is derived from `chunks_per_bin` and the bin's
    actual count. Per-bin mem is sized for the bin's max length (worst case)
    plus `mem_safety`. Per-bin runtime is `chunk_size × per_seq + startup
    overhead × time_safety`.
    """
    if not stats.lengths_sorted:
        raise ValueError(
            "compute_bin_plan needs stats.lengths_sorted — call compute_input_stats."
        )

    if mode == "quantile":
        eff = _quantile_thresholds(stats.lengths_sorted, num_bins)
    elif mode == "thresholds":
        if not thresholds:
            raise ValueError("mode=thresholds requires non-empty thresholds")
        eff = sorted(int(t) for t in thresholds)
    else:
        raise ValueError(f"Unknown bin mode: {mode!r}")

    buckets = _bin_lengths(stats.lengths_sorted, eff)

    chunks_per_bin = max(1, int(chunks_per_bin))

    bin_specs: list[BinSpec] = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            # Still emit a placeholder so the recipe length matches num_bins.
            bin_specs.append(BinSpec(
                bin_idx=i,
                num_seqs=0,
                len_lo=0, len_hi=0, repr_len=0,
                chunk_size=1,
                mem_mb=int(stage_models.get("per_gpu", {}).get(
                    stage_models.get("default_gpu", "h100"), {}
                ).get("mem_mb", {}).get("base", 16000)),
                runtime_min=max(1, target_chunk_runtime_min or 30),
            ))
            continue

        len_lo = bucket[0]
        len_hi = bucket[-1]
        # p95 within the bin — keeps sizing safe for the upper edge.
        p95_idx = min(len(bucket) - 1, max(0, int(round(0.95 * (len(bucket) - 1)))))
        repr_len = bucket[p95_idx]

        # Memory: size for worst-case length in the bin.
        mem_raw = _eval_mem(
            coeffs_per_gpu["mem_mb"],
            p95_len=len_hi,
            mean_len=sum(bucket) / len(bucket),
            total_residues=sum(bucket),
            count=len(bucket),
        )
        mem_mb = int(math.ceil(mem_raw * mem_safety))

        # Runtime per seq: pick the right coefficient block (handles ESMFold's
        # chunked-trunk regime when this bin is in the long tail).
        rt_block = coeffs_per_gpu.get("runtime_sec_per_seq")
        if stage == "esmfold":
            chunk_threshold = stage_models.get("chunk_threshold", 10**9)
            chunked_block = coeffs_per_gpu.get("runtime_sec_per_seq_chunked")
            if chunked_block is not None and len_hi >= chunk_threshold:
                rt_block = chunked_block

        if rt_block is not None:
            per_seq_sec = _eval_time_per_seq(rt_block, mean_len=repr_len)
            per_seq_sec = max(per_seq_sec, 0.5)
            if stage == "boltz":
                per_seq_sec *= (boltz_recycling * boltz_samples) / 250
        else:
            coeffs_t = coeffs_per_gpu.get("runtime_sec_per_chunk", {})
            per_seq_sec = (
                coeffs_t.get("per_seq", 0)
                + coeffs_t.get("per_residue", 0) * repr_len
            )

        # Chunk size for this bin: split bucket into `chunks_per_bin` chunks
        # (or fewer if the bin has fewer items than that).
        n = len(bucket)
        n_chunks_in_bin = min(chunks_per_bin, n)
        chunk_size = math.ceil(n / n_chunks_in_bin)

        # Runtime: per-seq cost × chunk_size + base setup cost (e.g. MSA's DB scan).
        runtime_sec = per_seq_sec * chunk_size * time_safety
        coeffs_t_extra = coeffs_per_gpu.get("runtime_sec_per_chunk", {})
        runtime_sec += coeffs_t_extra.get("base", 0)
        runtime_min = max(1, int(math.ceil(runtime_sec / 60))) + max(0, startup_overhead_min)

        bin_specs.append(BinSpec(
            bin_idx=i,
            num_seqs=n,
            len_lo=len_lo, len_hi=len_hi, repr_len=repr_len,
            chunk_size=chunk_size,
            mem_mb=mem_mb,
            runtime_min=runtime_min,
        ))

    return BinPlan(
        mode=mode,
        num_bins=len(eff) + 1,
        chunks_per_bin=chunks_per_bin,
        thresholds=list(eff),
        bins=bin_specs,
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
      <stage>.max_files_per_job  (all stages — chunkers split by files-per-job)
    Preserves all other keys. Returns the backup path (or original path).
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    slurm = cfg.setdefault("slurm", {})
    slurm_resources = slurm.setdefault("resources", {})

    for stage, est in estimates.items():
        if est.num_chunks == 0:
            continue
        # SLURM resources (safety-net fallback; bin-aware chunks override per-job)
        slurm_resources[stage] = {
            "mem_mb": est.mem_mb,
            "runtime": est.runtime_min,
            "cpus_per_task": est.cpus,
            "gpus": est.gpus,
        }
        # Stage partition (only set if estimator picked one)
        if est.partition:
            slurm.setdefault(stage, {})["partition"] = est.partition

        # Chunk size = files per job. Every chunker (MSA/Boltz/ESMC/ESMFold2)
        # splits by max_files_per_job now.
        stage_cfg = cfg.setdefault(stage, {})
        stage_cfg["max_files_per_job"] = est.chunk_size

        # If a BinPlan was produced, populate the binning recipe so the chunker
        # produces per-bin chunks with per-chunk SLURM resources at run time.
        if est.bin_plan is not None:
            stage_cfg = cfg.setdefault(stage, {})
            existing = stage_cfg.get("binning") or {}
            stage_cfg["binning"] = {
                "enabled": True,
                "mode": est.bin_plan.mode,
                "num_bins": est.bin_plan.num_bins,
                "chunks_per_bin": est.bin_plan.chunks_per_bin,
                # Echo computed thresholds for transparency (used in thresholds mode;
                # in quantile mode the chunker recomputes from input distribution).
                "thresholds": list(est.bin_plan.thresholds),
                "startup_overhead_min": existing.get("startup_overhead_min", 5),
                "bins": [
                    {
                        "mem_mb": b.mem_mb,
                        "runtime_min": b.runtime_min,
                    }
                    for b in est.bin_plan.bins
                ],
            }

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
