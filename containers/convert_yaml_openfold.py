from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Dict

import yaml


def parser_yaml(file_path: Path):
    """Function that parses a .yaml used for boltz, returns a dict."""

    with open(file_path, "r") as stream:
        data = yaml.safe_load(stream)

    return data


def write_openfold_schema(data: Dict, yaml_path: Path):

    openfold_schema = {"queries": {}}

    chains = []
    for chain_idx, entry in enumerate(data["sequences"]):
        prot_data = entry["protein"]
        chain_id = chr(ord("A") + chain_idx)
        has_msa = "msa" in prot_data

        of_prot = {
            "molecule_type": "protein",
            "chain_ids": chain_id,
            "sequence": prot_data["sequence"],
        }

        if has_msa:
            of_prot["main_msa_file_paths"] = str(Path(prot_data["msa"]).parent)

        chains.append(of_prot)

    openfold_schema["queries"] = {yaml_path.stem: {"chains": chains}}

    return openfold_schema


def write_json(schema: Dict, yaml_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / yaml_path.with_suffix(".json").name
    out_path.write_text(json.dumps(schema, indent=2) + "\n")
    return out_path


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)


    args = parser.parse_args()

    source_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) if args.out_dir else source_dir

    yamls = sorted(source_dir.glob("*.yaml"))
    for yaml_path in yamls:
        data = parser_yaml(yaml_path)

        openfold_schema = write_openfold_schema(data, yaml_path)
        print(json.dumps(openfold_schema, indent=2))

        out_path = write_json(openfold_schema, yaml_path, out_dir)
        print(f"Wrote {out_path}")

