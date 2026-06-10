"""
Results data layer for the ProtForge web UI.

Pure functions (no Streamlit deps) that read the pipeline's output tree and
benchmark TSVs into plain dicts the Results tab renders. Kept importable and
testable like estimator/validate.

Output tree (under output.parent_dir):
    sequences/{seq}/boltz/*_model_*.cif       + confidence_*_model_*.json
    sequences/{seq}/openfold/*_model.cif      + *_confidences_aggregated.json
    sequences/{seq}/esmfold/fast/structure.cif + plddt.npy
    benchmarks/{stage}/*.tsv                   (Snakemake benchmark format)
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

# Stage -> glob(s) for kept structure files, relative to sequences/{seq}/.
_STRUCTURE_GLOBS: dict[str, list[str]] = {
    "boltz": ["boltz/*_model_*.cif", "boltz/*_model_*.pdb"],
    "openfold": ["openfold/*_model.cif", "openfold/*_model.pdb",
                 "openfold/*_model.cif.gz"],
    "esmfold": ["esmfold/*/structure.cif", "esmfold/*/structure.pdb"],
}

# Snakemake benchmark stages we know how to label.
BENCHMARK_STAGES = ["msa", "boltz", "esmc", "esmc_sae", "esmfold", "openfold"]


# --- Structures -----------------------------------------------------------


@dataclass
class StructureFile:
    stage: str
    path: Path
    label: str  # display label, e.g. "boltz / mutant_A_model_0.cif"


def sequences_root(output_dir: str | Path) -> Path:
    return Path(output_dir) / "sequences"


def list_sequence_dirs(output_dir: str | Path) -> list[str]:
    """All sequence dir names under sequences/ — a single iterdir, no per-dir
    globbing. Cheap enough to call on every rerun even for large runs; use this
    to populate the picker and check structures lazily for the chosen one."""
    root = sequences_root(output_dir)
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def list_result_sequences(output_dir: str | Path) -> list[str]:
    """Names of sequence dirs that contain at least one predicted structure.

    Globs every sequence — O(N × stages). Prefer list_sequence_dirs() for the
    UI picker and resolve structures lazily; this stays for callers that need
    the filtered set (and the tests)."""
    root = sequences_root(output_dir)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and find_structures(d):
            out.append(d.name)
    return out


def find_structures(seq_dir: str | Path) -> list[StructureFile]:
    """All kept structure files for one sequence dir, across stages."""
    seq_dir = Path(seq_dir)
    found: list[StructureFile] = []
    for stage, globs in _STRUCTURE_GLOBS.items():
        for pattern in globs:
            for p in sorted(seq_dir.glob(pattern)):
                if p.is_file():
                    found.append(StructureFile(
                        stage=stage, path=p, label=f"{stage} / {p.name}"))
    return found


def read_structure_text(path: str | Path) -> str:
    """Return the structure file's text, transparently decompressing .gz."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return f.read()
    return path.read_text(errors="replace")


def structure_format(path: str | Path) -> str:
    """'cif' or 'pdb' for a (possibly .gz) structure path — for the 3D viewer."""
    name = Path(path).name.lower()
    if ".cif" in name:
        return "cif"
    if ".pdb" in name:
        return "pdb"
    return "cif"


# --- Confidence -----------------------------------------------------------


def read_confidence(structure: StructureFile) -> dict[str, float]:
    """Best-effort per-model confidence scalars for a structure.

    Boltz/OpenFold write sibling confidence JSONs; ESMFold writes plddt.npy.
    Returns a flat {metric: value} dict (empty if nothing parseable found).
    """
    p = structure.path
    stage = structure.stage

    if stage == "esmfold":
        return _read_esmfold_plddt(p.parent)

    # Boltz / OpenFold: find a sibling JSON sharing the model stem.
    stem = p.name
    for suffix in ("_model.cif.gz", "_model.cif", "_model.pdb"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    else:
        # Boltz stems look like "<seq>_model_0.cif"
        stem = p.stem

    metrics: dict[str, float] = {}
    for js in sorted(p.parent.glob("*.json")):
        name = js.name.lower()
        if "confidence" not in name and "ranking" not in name:
            continue
        # Prefer JSONs that reference this model's stem; fall back to any.
        if stem and stem.split("_model")[0] not in js.name:
            continue
        metrics.update(_flatten_scalar_json(js))
    return metrics


def _flatten_scalar_json(path: Path) -> dict[str, float]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    out: dict[str, float] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out[k] = float(v)
    return out


def _read_esmfold_plddt(fast_dir: Path) -> dict[str, float]:
    npy = fast_dir / "plddt.npy"
    if not npy.is_file():
        return {}
    try:
        import numpy as np
        arr = np.load(npy)
        return {"mean_plddt": float(arr.mean()), "min_plddt": float(arr.min())}
    except Exception:
        return {}


# --- Benchmarks -----------------------------------------------------------


@dataclass
class StageBenchmark:
    stage: str
    n_jobs: int
    total_s: float
    mean_s: float
    p95_s: float
    max_s: float
    max_rss_mb: float        # peak across jobs
    node_hours: float        # sum of wall-clock, hours
    runtimes_s: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "runtimes_s"}
        return d


def benchmarks_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "benchmarks"


def read_benchmarks(output_dir: str | Path) -> dict[str, StageBenchmark]:
    """Parse Snakemake benchmark TSVs into per-stage aggregates.

    Each stage dir holds one TSV per rule invocation with columns including
    's' (wall-clock seconds) and 'max_rss' (MB). Stages with no TSVs are
    omitted. Returns {stage: StageBenchmark}.
    """
    bench = benchmarks_dir(output_dir)
    if not bench.is_dir():
        return {}

    per_stage: dict[str, list[tuple[float, float]]] = {}
    for stage_dir in sorted(bench.iterdir()):
        if not stage_dir.is_dir():
            continue
        stage = stage_dir.name
        rows: list[tuple[float, float]] = []
        for tsv in stage_dir.glob("*.tsv"):
            try:
                with open(tsv) as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        s = float(row.get("s", 0) or 0)
                        rss = row.get("max_rss") or row.get("max_uss") or 0
                        rows.append((s, float(rss) if rss else 0.0))
            except Exception:
                continue
        if rows:
            per_stage[stage] = rows

    out: dict[str, StageBenchmark] = {}
    for stage, rows in per_stage.items():
        times = sorted(r[0] for r in rows)
        rss = [r[1] for r in rows]
        n = len(times)
        p95_idx = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
        out[stage] = StageBenchmark(
            stage=stage,
            n_jobs=n,
            total_s=sum(times),
            mean_s=sum(times) / n,
            p95_s=times[p95_idx],
            max_s=times[-1],
            max_rss_mb=max(rss) if rss else 0.0,
            node_hours=sum(times) / 3600.0,
            runtimes_s=times,
        )
    return out
