#!/usr/bin/env python3
"""
Organize OpenFold3 prediction outputs into the canonical
sequences/{seq}/openfold/ layout (mirrors organize_boltz_outputs.py).

OpenFold writes, per query (== Boltz YAML stem == sequence name):

    <output_dir>/{query}/seed_{S}/
        {query}_seed_{S}_sample_{n}_model.cif
        {query}_seed_{S}_sample_{n}_confidences.json
        {query}_seed_{S}_sample_{n}_confidences_aggregated.json
        timing.json

We rank a query's samples (across all seeds) by `sample_ranking_score` from the
aggregated confidence JSON, keep the top-N (or all), and copy each kept sample's
.cif + confidence JSONs to sequences/{seq}/openfold/.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Per-query metadata files OpenFold drops next to the query dirs (not queries).
_METADATA_NAMES = {
    "inference_query_set.json",
    "model_config.json",
    "experiment_config.json",
}

_MODEL_SUFFIXES = ("_model.cif", "_model.pdb", "_model.cif.gz")


def _sample_prefix(model_file: Path) -> str:
    """`foo_seed_42_sample_1_model.cif` -> `foo_seed_42_sample_1`."""
    name = model_file.name
    for suf in _MODEL_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return model_file.stem


def _ranking_score(agg_json: Path) -> float:
    try:
        data = json.loads(agg_json.read_text())
    except (OSError, json.JSONDecodeError):
        return float("-inf")
    for key in ("sample_ranking_score", "ranking_score", "ptm", "avg_plddt"):
        val = data.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return float("-inf")


def _collect_samples(query_dir: Path) -> list[dict]:
    """Return one record per sample: model file, sibling files, ranking score."""
    samples = []
    for seed_dir in sorted(query_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        for model_file in sorted(seed_dir.iterdir()):
            if not (model_file.is_file()
                    and any(model_file.name.endswith(s) for s in _MODEL_SUFFIXES)):
                continue
            prefix = _sample_prefix(model_file)
            files = [f for f in seed_dir.iterdir()
                     if f.is_file() and f.name.startswith(prefix)]
            agg = next((f for f in files
                        if f.name.endswith("_confidences_aggregated.json")), None)
            samples.append({
                "prefix": prefix,
                "files": files,
                "score": _ranking_score(agg) if agg else float("-inf"),
            })
    return samples


def _target_dir(seq_base: Path, seq_id: str) -> Path:
    """sequences/{seq}/openfold/, honoring the seq_ prefix convention."""
    for candidate in (f"seq_{seq_id}", seq_id):
        if (seq_base / candidate).exists():
            return seq_base / candidate / "openfold"
    return seq_base / f"seq_{seq_id}" / "openfold"


def organize(openfold_output_dir: str, sequences_dir: str,
             samples_to_save: "int | str") -> int:
    out_path = Path(openfold_output_dir)
    seq_base = Path(sequences_dir)

    processed, skipped = 0, 0
    for query_dir in sorted(out_path.iterdir()):
        if not query_dir.is_dir() or query_dir.name in _METADATA_NAMES:
            continue
        seq_id = query_dir.name
        samples = _collect_samples(query_dir)
        if not samples:
            print(f"WARNING: no model files for {seq_id}")
            skipped += 1
            continue

        samples.sort(key=lambda s: s["score"], reverse=True)
        keep = samples if samples_to_save == "all" else samples[:int(samples_to_save)]

        target = _target_dir(seq_base, seq_id)
        target.mkdir(parents=True, exist_ok=True)
        # Drop previously organized artifacts so reruns don't leave stale samples.
        for existing in target.iterdir():
            if existing.is_file():
                existing.unlink()

        copied = 0
        for sample in keep:
            for f in sample["files"]:
                shutil.copy2(f, target / f.name)
                copied += 1
        label = "all" if samples_to_save == "all" else f"top {len(keep)}"
        print(f"Organized {seq_id}: {label} sample(s) ({copied} files) -> {target}")
        processed += 1

    print(f"Processed: {processed}, Skipped: {skipped}")
    return processed


def main():
    p = argparse.ArgumentParser(description="Organize OpenFold3 outputs")
    p.add_argument("--openfold_output_dir", required=True)
    p.add_argument("--sequences_dir", required=True)
    p.add_argument("--samples_to_save", default="1",
                   help='Top-N samples to keep (int >= 1) or "all".')
    args = p.parse_args()

    if args.samples_to_save == "all":
        samples_to_save: "int | str" = "all"
    else:
        try:
            samples_to_save = int(args.samples_to_save)
            if samples_to_save < 1:
                raise ValueError
        except ValueError:
            p.error("--samples_to_save must be a positive int or 'all'")

    if organize(args.openfold_output_dir, args.sequences_dir, samples_to_save) == 0:
        print("ERROR: No predictions were organized", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
