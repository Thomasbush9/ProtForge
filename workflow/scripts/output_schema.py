#!/usr/bin/env python3
"""
Normalized prediction-output contract shared by the structure predictors.

Boltz, OpenFold3, and ESMFold2 each write confidence in their own shape (per-
model JSON, aggregated JSON, plddt.npy + metrics.pt) and name their structure
files differently. This module defines ONE schema and the extraction that maps
each predictor onto it, so downstream consumers (the webapp Results tab, and
later analyses) read a single format instead of special-casing each predictor.

The organize_* scripts call `extract_summary` + `write_summary` to drop a
`<model_id>.summary.json` sidecar next to each kept structure. The webapp calls
`summary_for_structure`, which prefers the sidecar and falls back to live
extraction for outputs produced before sidecars existed — identical result
either way, since both go through `extract_summary`.

Schema (`<model_id>.summary.json`):
    schema_version : int
    stage          : "boltz" | "openfold" | "esmfold"
    model_id       : str   e.g. "model_0", "seed_42_sample_0", "fast"
    structure      : str   structure filename (sits in the same dir)
    plddt_mean     : float | null   normalized to 0-100
    ptm            : float | null
    iptm           : float | null
    ranking_score  : float | null   predictor's own overall score
    plddt_in_bfactor : bool | null  whether the structure's B-factor column
                                    carries pLDDT (so the viewer can colour by
                                    it); null when it couldn't be determined
    metrics        : dict  all raw scalar metrics found, for transparency
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

SCHEMA_VERSION = 1
SUMMARY_SUFFIX = ".summary.json"

_MODEL_SUFFIXES = ("_model.cif.gz", "_model.cif", "_model.pdb",
                   ".cif.gz", ".cif", ".pdb")


@dataclass
class ModelSummary:
    schema_version: int
    stage: str
    model_id: str
    structure: str
    plddt_mean: float | None
    ptm: float | None
    iptm: float | None
    ranking_score: float | None
    plddt_in_bfactor: bool | None
    metrics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# --- small helpers --------------------------------------------------------


def _read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return f.read()
    return path.read_text(errors="replace")


def _norm_plddt(v: float | None) -> float | None:
    """Normalize a pLDDT to the 0-100 convention (predictors vary 0-1 vs 0-100)."""
    if v is None:
        return None
    v = float(v)
    return v * 100.0 if v <= 1.0 else v


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


def model_id_for(stage: str, structure_path: Path) -> str:
    """Stable per-model id used as the sidecar filename stem."""
    name = structure_path.name
    if stage == "boltz":
        m = re.search(r"(model_\d+)", name)
        if m:
            return m.group(1)
    elif stage == "openfold":
        m = re.search(r"seed_(\d+)_sample_(\d+)", name)
        if m:
            return f"seed_{m.group(1)}_sample_{m.group(2)}"
    elif stage == "esmfold":
        # esmfold/{variant}/structure.cif -> the variant dir name ("fast")
        return structure_path.parent.name
    # Fallback: filename without known structure suffixes
    for suf in _MODEL_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return structure_path.stem


# --- B-factor / pLDDT detection -------------------------------------------


def structure_plddt_in_bfactor(structure_path: Path, sample: int = 2000) -> bool | None:
    """Best-effort: does the structure's B-factor column carry pLDDT?

    Returns True if B-factors vary and sit in a plausible pLDDT range (0-100,
    not all ~0), False if they're constant/zero or out of range, None if the
    column couldn't be parsed. Bounded to the first `sample` atoms for speed.
    """
    try:
        text = _read_text(structure_path)
    except Exception:
        return None
    name = structure_path.name.lower()
    values = (_pdb_bfactors(text, sample) if ".pdb" in name
              else _cif_bfactors(text, sample))
    if not values:
        return None
    distinct = {round(v, 2) for v in values}
    if len(distinct) <= 1:
        return False
    mn, mx = min(values), max(values)
    return bool(0.0 <= mn and 1.0 < mx <= 100.0)


def _pdb_bfactors(text: str, sample: int) -> list[float]:
    out: list[float] = []
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 66:
            try:
                out.append(float(line[60:66]))
            except ValueError:
                pass
            if len(out) >= sample:
                break
    return out


def _cif_bfactors(text: str, sample: int) -> list[float]:
    """Parse _atom_site.B_iso_or_equiv from the atom_site loop of an mmCIF."""
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() == "loop_":
            headers: list[str] = []
            j = i + 1
            while j < n and lines[j].lstrip().startswith("_"):
                headers.append(lines[j].strip())
                j += 1
            if any(h.startswith("_atom_site.") for h in headers):
                try:
                    b_idx = headers.index("_atom_site.B_iso_or_equiv")
                except ValueError:
                    return []
                out: list[float] = []
                k = j
                while k < n:
                    row = lines[k].strip()
                    if not row or row.startswith(("_", "#", "loop_")):
                        break
                    cols = row.split()
                    if len(cols) > b_idx:
                        try:
                            out.append(float(cols[b_idx]))
                        except ValueError:
                            pass
                    if len(out) >= sample:
                        break
                    k += 1
                return out
            i = j
        else:
            i += 1
    return []


# --- per-stage extraction -------------------------------------------------


def _sibling_json(structure_path: Path, *, must_contain: tuple[str, ...],
                  prefer_suffix: str | None = None) -> Path | None:
    """Find a confidence JSON next to the structure that shares its model stem.

    The structure stem (filename minus the model/structure suffix) is a
    substring of its confidence JSON's name for every predictor:
      boltz:    34073_model_0(.cif)       -> confidence_34073_model_0.json
      openfold: ..._sample_9(_model.cif)  -> ..._sample_9_confidences*.json
    Match on the *full* stem so per-model files stay distinct — matching only
    the sequence prefix would cross-wire model_0/model_1 (or sample_8/sample_9)
    and attach the wrong confidence to a structure.
    """
    stem = structure_path.name
    for suf in _MODEL_SUFFIXES:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    candidates = [p for p in structure_path.parent.glob("*.json")
                  if any(t in p.name.lower() for t in must_contain)
                  and stem in p.name]
    if not candidates:
        return None
    if prefer_suffix:
        preferred = [p for p in candidates if p.name.endswith(prefer_suffix)]
        if preferred:
            return preferred[0]
    return sorted(candidates)[0]


def extract_summary(stage: str, structure_path: Path) -> ModelSummary:
    """Build a ModelSummary for one structure by reading its native confidence."""
    structure_path = Path(structure_path)
    metrics: dict[str, float] = {}
    plddt = ptm = iptm = ranking = None

    if stage == "boltz":
        js = _sibling_json(structure_path, must_contain=("confidence",))
        if js:
            metrics = _flatten_scalar_json(js)
        ptm = metrics.get("ptm")
        iptm = metrics.get("iptm")
        ranking = metrics.get("confidence_score")
        plddt = _norm_plddt(metrics.get("complex_plddt", metrics.get("plddt")))

    elif stage == "openfold":
        js = _sibling_json(structure_path, must_contain=("confidence", "ranking"),
                           prefer_suffix="_confidences_aggregated.json")
        if js:
            metrics = _flatten_scalar_json(js)
        ptm = metrics.get("ptm")
        iptm = metrics.get("iptm")
        ranking = metrics.get("ranking_score", metrics.get("sample_ranking_score"))
        plddt = _norm_plddt(metrics.get("avg_plddt", metrics.get("mean_plddt")))

    elif stage == "esmfold":
        metrics, plddt, ptm = _esmfold_metrics(structure_path.parent)
        ranking = None
        iptm = None

    return ModelSummary(
        schema_version=SCHEMA_VERSION,
        stage=stage,
        model_id=model_id_for(stage, structure_path),
        structure=structure_path.name,
        plddt_mean=plddt,
        ptm=ptm,
        iptm=iptm,
        ranking_score=ranking,
        plddt_in_bfactor=structure_plddt_in_bfactor(structure_path),
        metrics=metrics,
    )


def _esmfold_metrics(fast_dir: Path) -> tuple[dict, float | None, float | None]:
    metrics: dict[str, float] = {}
    plddt = None
    ptm = None

    # Prefer the torch-free metrics.json the runner writes (ptm, mean_plddt);
    # fall back to plddt.npy (numpy) and finally metrics.pt (torch) for runs
    # produced before the JSON existed.
    js = fast_dir / "metrics.json"
    if js.is_file():
        for k, v in _flatten_scalar_json(js).items():
            metrics[k] = v
        if "ptm" in metrics:
            ptm = metrics["ptm"]
        if "mean_plddt" in metrics:
            plddt = _norm_plddt(metrics["mean_plddt"])

    npy = fast_dir / "plddt.npy"
    if plddt is None and npy.is_file():
        try:
            import numpy as np
            arr = np.load(npy)
            metrics["mean_plddt"] = float(arr.mean())
            metrics["min_plddt"] = float(arr.min())
            plddt = _norm_plddt(metrics["mean_plddt"])
        except Exception:
            pass

    pt = fast_dir / "metrics.pt"
    if ptm is None and pt.is_file():
        try:
            import torch
            meta = torch.load(pt, map_location="cpu", weights_only=False)
            if isinstance(meta, dict) and isinstance(meta.get("ptm"), (int, float)):
                ptm = float(meta["ptm"])
                metrics["ptm"] = ptm
        except Exception:
            pass
    return metrics, plddt, ptm


# --- sidecar I/O ----------------------------------------------------------


def sidecar_path(target_dir: Path, model_id: str) -> Path:
    return Path(target_dir) / f"{model_id}{SUMMARY_SUFFIX}"


def write_summary(summary: ModelSummary, target_dir: Path) -> Path:
    """Write a summary sidecar into target_dir; returns its path."""
    path = sidecar_path(target_dir, summary.model_id)
    path.write_text(json.dumps(summary.as_dict(), indent=2))
    return path


def write_summary_for(stage: str, structure_path: Path) -> Path:
    """Extract + write a sidecar next to the given structure. Convenience for
    the organize_* scripts."""
    structure_path = Path(structure_path)
    summary = extract_summary(stage, structure_path)
    return write_summary(summary, structure_path.parent)


def summary_for_structure(stage: str, structure_path: Path) -> dict:
    """Return the normalized summary dict for a structure, preferring an existing
    sidecar and falling back to live extraction. Used by the webapp."""
    structure_path = Path(structure_path)
    side = sidecar_path(structure_path.parent, model_id_for(stage, structure_path))
    if side.is_file():
        try:
            return json.loads(side.read_text())
        except Exception:
            pass
    return extract_summary(stage, structure_path).as_dict()
