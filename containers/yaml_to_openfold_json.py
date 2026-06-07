#!/usr/bin/env python3
"""Convert Boltz-format YAML inputs to an OpenFold3 query JSON (+ runner.yml)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

DEFAULT_MAX_SEQ_COUNT = 10_000


def _resolve_msa(msa_value: str, yaml_path: Path) -> Path | None:
    value = str(msa_value).strip().strip('"').strip("'")
    if not value or value.lower() == "empty":
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (yaml_path.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{yaml_path}: MSA not found: {path}")
    return path


def _chain_from_protein(protein: dict, yaml_path: Path, container_prefix: str) -> tuple[dict, set[str]]:
    if "sequence" not in protein:
        raise KeyError(f"{yaml_path}: protein entry missing 'sequence'")
    chain_id = str(protein.get("id", "A")).strip().strip('"').strip("'")
    chain: dict = {
        "molecule_type": "protein",
        "chain_ids": [chain_id],
        "sequence": str(protein["sequence"]).strip(),
    }
    stems: set[str] = set()
    if "msa" in protein:
        msa_path = _resolve_msa(str(protein["msa"]), yaml_path)
        if msa_path is not None:
            container_msa = f"{container_prefix}/msa/{msa_path.name}"
            chain["main_msa_file_paths"] = [container_msa]
            stems.add(msa_path.stem)
    return chain, stems


def yaml_to_query(yaml_path: Path, container_prefix: str) -> tuple[str, list[dict], set[str], dict[Path, Path]]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    if not data or "sequences" not in data:
        raise KeyError(f"{yaml_path}: missing 'sequences'")

    chains: list[dict] = []
    stems: set[str] = set()
    msa_links: dict[Path, Path] = {}
    for entry in data["sequences"]:
        if "protein" not in entry:
            raise KeyError(f"{yaml_path}: expected protein entries under 'sequences'")
        chain, chain_stems = _chain_from_protein(entry["protein"], yaml_path, container_prefix)
        chains.append(chain)
        stems |= chain_stems
        if "msa" in entry["protein"]:
            host_msa = _resolve_msa(str(entry["protein"]["msa"]), yaml_path)
            if host_msa is not None:
                msa_links[host_msa] = Path("msa") / host_msa.name

    return yaml_path.stem, chains, stems, msa_links


def build_runner_yml(stems: set[str], max_seq_count: int) -> dict:
    ordered = sorted(stems)
    return {
        "dataset_config_kwargs": {
            "msa": {
                "max_seq_counts": {stem: max_seq_count for stem in ordered},
                "aln_order": ordered,
            }
        }
    }


def convert_input_dir(
    input_dir: Path,
    work_dir: Path,
    container_prefix: str,
    max_seq_count: int,
) -> tuple[Path, Path | None]:
    yaml_files = sorted(input_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No *.yaml in {input_dir}")

    queries: dict[str, dict] = {}
    all_stems: set[str] = set()
    all_msa_links: dict[Path, Path] = {}

    for yaml_path in yaml_files:
        query_name, chains, stems, msa_links = yaml_to_query(yaml_path, container_prefix)
        if query_name in queries:
            raise ValueError(f"Duplicate query name {query_name!r} in {input_dir}")
        queries[query_name] = {"chains": chains}
        all_stems |= stems
        all_msa_links.update(msa_links)

    work_dir.mkdir(parents=True, exist_ok=True)
    msa_dir = work_dir / "msa"
    msa_dir.mkdir(exist_ok=True)
    for host_path, rel_path in all_msa_links.items():
        dest = work_dir / rel_path
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(host_path)

    query_json = work_dir / "query.json"
    query_json.write_text(json.dumps({"queries": queries}, indent=2) + "\n")

    runner_path: Path | None = None
    if all_stems:
        runner_path = work_dir / "inference_precomputed.yml"
        import yaml as yaml_mod

        runner_path.write_text(
            yaml_mod.dump(build_runner_yml(all_stems, max_seq_count), sort_keys=False)
        )

    return query_json, runner_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of Boltz *.yaml")
    parser.add_argument("--work-dir", type=Path, required=True, help="Writable dir for query.json and msa symlinks")
    parser.add_argument(
        "--container-prefix",
        default="/data",
        help="Path prefix inside the container (default: /data)",
    )
    parser.add_argument(
        "--max-seq-count",
        type=int,
        default=DEFAULT_MAX_SEQ_COUNT,
        help="max_seq_counts per MSA stem in runner.yml",
    )
    args = parser.parse_args()

    query_json, runner_path = convert_input_dir(
        args.input_dir.resolve(),
        args.work_dir.resolve(),
        args.container_prefix.rstrip("/"),
        args.max_seq_count,
    )
    print(f"Wrote {query_json}")
    if runner_path:
        print(f"Wrote {runner_path}")
    else:
        print("No MSAs: inference_precomputed.yml not written (MSA-free run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
