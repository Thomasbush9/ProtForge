'''Script to build jsons for openfold format'''
from pathlib import Path 
from argparse import ArgumentParser
from typing import List, Dict 
import yaml
import shutil
import json 

def parser_yaml(file_path:Path):
    with open(file_path, "r") as stream: 
        data = yaml.safe_load(stream)

    return data


def write_openfold_schema(data:Dict):
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
    return chains

def rename_msa_file(old_msa_path): 
    old_msa_path = Path(old_msa_path)

    if old_msa_path.name != "uniref90_hits.a3m":
        new_path = old_msa_path.with_name("uniref90_hits.a3m")
        return new_path
    return old_msa_path


def write_batch_openfold_schema(batch:List[Path]):
    
    #TODO: add assert to be only .yamls passed 
    openfold_schema = {"queries":{}}

    for file_path in batch: 
        data = parser_yaml(file_path)
        try:
            msa_path = data["sequences"][0]["protein"]["msa"]
            new_msa_path = rename_msa_file(msa_path)
            shutil.copy2(src=msa_path, dst=str(new_msa_path))
        except FileNotFoundError:
            continue
        finally:
            openfold_schema["queries"][file_path.stem] = {"chains": write_openfold_schema(data)}

    return openfold_schema
        
def split_dir_json(yamls:List, batch_size:int): 
    collection = []
    start = 0 
    while start < len(yamls):
        end = start + batch_size
        openfold_json = write_batch_openfold_schema(yamls[start:end])
        start += batch_size
        collection.append(openfold_json)
    
    return collection

def write_json(schema: Dict, yaml_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / yaml_path.with_suffix(".json").name
    out_path.write_text(json.dumps(schema, indent=2) + "\n")
    return out_path


if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("--input", help="Input diretory")
    parser.add_argument("--out", help="Output directory")
    parser.add_argument("--batch_size", type=int,default=32, help="How many files process per node")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.out)
    batch_size = args.batch_size

    # get all the files to convert: 
    yamls = [file for file in input_dir.rglob("*.yaml")]
    print(len(yamls))
    print(f"Converting {len(yamls)} in {len(yamls)//batch_size} Batch size: {batch_size}")
    collection_json = split_dir_json(yamls, batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, dict_file in enumerate(collection_json):
        out_path = output_dir / f"openfold_json_{i}.json"
        out_path.write_text(json.dumps(dict_file, indent=2) + "\n")






