#!/usr/bin/env python3
"""
Prepare OpenFold3 chunk JSONs from Boltz-style YAMLs.

Each chunk = one OpenFold query JSON = one SLURM job (the job batches all its
queries across its GPUs via PyTorch Lightning). This mirrors
prepare_boltz_chunks.py but emits JSON instead of symlinked YAML dirs.

Steps:
  1. Find .yaml files (sequences/{name}/{name}.yaml after MSA, or user yaml_dir).
  2. Split into batches: exactly --num_batches batches if given, else
     --max_files_per_job queries per batch.
  3. Convert each batch YAML -> an OpenFold query (chains + main_msa_file_paths)
     and copy each sequence's a3m to the OpenFold-expected uniref90_hits.a3m.
  4. Write openfold_chunks/chunk_{i}.json + manifest.txt + a shared runner.yaml.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

# OpenFold only accepts MSAs under fixed filenames in main_msa_file_paths.
OPENFOLD_MSA_NAME = "uniref90_hits.a3m"


def find_yaml_files(search_dir: str) -> list[Path]:
    return sorted(Path(search_dir).rglob("*.yaml"))


def _prepare_msa_dir(msa_value) -> str | None:
    """Copy the sequence a3m to <dir>/uniref90_hits.a3m and return that dir.

    Boltz YAMLs store `msa:` as an absolute a3m path, or the literal "empty"
    when no homologs were found. Returns None (no MSA for this chain) when the
    value is empty/missing/unreadable, so the query still folds single-sequence.
    """
    if not msa_value or msa_value == "empty":
        return None
    src = Path(msa_value)
    if not src.is_file():
        return None
    if src.name != OPENFOLD_MSA_NAME:
        dst = src.with_name(OPENFOLD_MSA_NAME)
        shutil.copy2(src, dst)
    return str(src.parent)


def yaml_to_query(yaml_path: Path) -> dict:
    """Convert one Boltz YAML into an OpenFold query: {"chains": [...]}."""
    data = yaml.safe_load(yaml_path.read_text())
    chains = []
    for idx, entry in enumerate(data.get("sequences", []) or []):
        prot = entry.get("protein") or {}
        chain = {
            "molecule_type": "protein",
            "chain_ids": chr(ord("A") + idx),
            "sequence": prot["sequence"],
        }
        msa_dir = _prepare_msa_dir(prot.get("msa"))
        if msa_dir:
            chain["main_msa_file_paths"] = msa_dir
        chains.append(chain)
    return {"chains": chains}


def split_batches(yaml_files: list[Path], num_batches: int | None,
                  max_files_per_job: int) -> list[list[Path]]:
    """Split into exactly num_batches even batches, else max_files_per_job each."""
    total = len(yaml_files)
    if num_batches and num_batches > 0:
        n = min(num_batches, total)
        # Even-as-possible split into n contiguous batches.
        base, extra = divmod(total, n)
        batches, start = [], 0
        for i in range(n):
            size = base + (1 if i < extra else 0)
            batches.append(yaml_files[start:start + size])
            start += size
        return batches
    size = max(1, max_files_per_job)
    return [yaml_files[i:i + size] for i in range(0, total, size)]


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def write_runner_yaml(path: Path, *, devices: int, num_workers: int,
                      seeds, structure_format: str, write_full_confidence: bool,
                      recycling_iters, advanced: dict) -> Path:
    """Write runner.yaml from config. `advanced` is deep-merged last (escape
    hatch). recycling_iters, when set, goes to model_update.custom."""
    cfg: dict = {
        "pl_trainer_args": {
            "accelerator": "gpu",
            "devices": int(devices),
            "num_nodes": 1,
        },
        "data_module_args": {"num_workers": int(num_workers)},
        "experiment_settings": {"mode": "predict", "use_msa_server": False},
        "output_writer_settings": {
            "structure_format": structure_format,
            "write_full_confidence_scores": bool(write_full_confidence),
        },
    }
    if seeds:
        cfg["experiment_settings"]["seeds"] = list(seeds)
    if recycling_iters is not None:
        cfg.setdefault("model_update", {}).setdefault("custom", {})[
            "num_recycling_iters"
        ] = int(recycling_iters)
    if advanced:
        _deep_merge(cfg, advanced)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def main():
    p = argparse.ArgumentParser(description="Prepare OpenFold3 chunk JSONs")
    p.add_argument("--yaml_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_batches", type=int, default=None,
                   help="Split into exactly this many JSONs/jobs. Overrides "
                        "--max_files_per_job when set.")
    p.add_argument("--max_files_per_job", type=int, default=25,
                   help="Queries per JSON when --num_batches is unset.")
    # runner.yaml params
    p.add_argument("--devices", type=int, default=1, help="GPUs per job (1..4).")
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--seeds", default="", help="Comma list, e.g. 42,43.")
    p.add_argument("--structure_format", default="cif")
    p.add_argument("--write_full_confidence", action="store_true")
    p.add_argument("--recycling_iters", type=int, default=None)
    p.add_argument("--runner_advanced_json", default="",
                   help="JSON dict deep-merged into runner.yaml.")
    args = p.parse_args()

    yaml_files = find_yaml_files(args.yaml_dir)
    if not yaml_files:
        print(f"ERROR: No .yaml files found in {args.yaml_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(yaml_files)} YAML files in {args.yaml_dir}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Clean stale chunk JSONs from previous runs.
    for stale in out.glob("chunk_*.json"):
        stale.unlink()

    batches = split_batches(yaml_files, args.num_batches, args.max_files_per_job)

    manifest_lines = []
    for i, batch in enumerate(batches):
        if not batch:
            continue
        queries = {}
        for yaml_path in batch:
            queries[yaml_path.stem] = yaml_to_query(yaml_path)
        chunk_path = out / f"chunk_{i}.json"
        chunk_path.write_text(json.dumps({"queries": queries}, indent=2) + "\n")
        manifest_lines.append(str(chunk_path.resolve()))
        print(f"Wrote {chunk_path.name} with {len(queries)} queries")

    (out / "manifest.txt").write_text("\n".join(manifest_lines) + "\n")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else []
    advanced = json.loads(args.runner_advanced_json) if args.runner_advanced_json else {}
    runner_path = write_runner_yaml(
        out / "runner.yaml",
        devices=args.devices,
        num_workers=args.num_workers,
        seeds=seeds,
        structure_format=args.structure_format,
        write_full_confidence=args.write_full_confidence,
        recycling_iters=args.recycling_iters,
        advanced=advanced,
    )
    print(f"Created {len(manifest_lines)} chunks in {out}")
    print(f"Manifest:    {out / 'manifest.txt'}")
    print(f"runner.yaml: {runner_path}")


if __name__ == "__main__":
    main()
