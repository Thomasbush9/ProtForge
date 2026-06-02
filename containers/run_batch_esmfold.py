import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

ESMFOLD2_REPO = "biohub/ESMFold2-Fast"
ESMFOLD_VARIANT = "fast"


def _enforce_offline(cache_dir: str | None) -> str | None:
    if not cache_dir:
        return None
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return f"{cache_dir}/hub"


def parse_inputs(input_dir: str) -> dict[str, str]:
    paths = sorted(Path(input_dir).glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No *.yaml in {input_dir}")

    out: dict[str, str] = {}
    for p in paths:
        with open(p) as f:
            data = yaml.safe_load(f)
        if not data or "sequences" not in data:
            raise KeyError(f"{p}: missing 'sequences'")
        entry = data["sequences"][0]
        if "protein" not in entry or "sequence" not in entry["protein"]:
            raise KeyError(f"{p}: expected sequences[0].protein.sequence")
        stem = p.stem
        if stem in out:
            raise ValueError(f"Duplicate stem {stem!r} in {input_dir}")
        out[stem] = str(entry["protein"]["sequence"]).strip()
    return out


def _infer_output_to_pdb(model: ESMFold2Model, infer_output) -> str:
    if hasattr(model, "output_to_pdb"):
        pdbs = model.output_to_pdb(infer_output)
    else:
        pdbs = ESMFold2Model.output_to_pdb(infer_output)
    return pdbs[0] if isinstance(pdbs, list) else pdbs


def save_outputs(
    output_dir: str,
    name: str,
    variant: str,
    model: ESMFold2Model,
    infer_output,
) -> None:
    out_dir = Path(output_dir) / name / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_str = _infer_output_to_pdb(model, infer_output)
    (out_dir / "structure.pdb").write_text(pdb_str)

    plddt = infer_output["plddt"]
    if torch.is_tensor(plddt):
        plddt = plddt.detach().cpu().numpy()
    np.save(out_dir / "plddt.npy", plddt)

    ptm = infer_output.get("ptm")
    if torch.is_tensor(ptm):
        ptm = float(ptm.detach().cpu().mean())
    torch.save(
        {"name": name, "variant": variant, "ptm": ptm},
        out_dir / "outputs.pt",
    )


def load_model(hub_cache: str) -> ESMFold2Model:
    return ESMFold2Model.from_pretrained(
        ESMFOLD2_REPO,
        cache_dir=hub_cache,
        local_files_only=True,
    ).cuda().eval()


def fold_sequence(
    model: ESMFold2Model,
    sequence: str,
    num_loops: int,
    num_sampling_steps: int,
):
    with torch.inference_mode():
        return model.infer_protein(
            sequence,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-loops", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    args = parser.parse_args()
    hub_cache = _enforce_offline(args.cache)

    seq_by_name = parse_inputs(args.input_dir)
    model = load_model(hub_cache)

    for name, sequence in seq_by_name.items():
        print(f"Folding {name} ({len(sequence)} aa)...", flush=True)
        out = fold_sequence(
            model,
            sequence,
            args.num_loops,
            args.num_sampling_steps,
        )
        save_outputs(args.output_dir, name, ESMFOLD_VARIANT, model, out)
        plddt = out["plddt"]
        plddt_mean = float(plddt.mean()) if torch.is_tensor(plddt) else float(np.mean(plddt))
        ptm = out.get("ptm")
        ptm_mean = float(ptm.mean()) if torch.is_tensor(ptm) else float(ptm)
        print(f"  {name}: pLDDT mean {plddt_mean:.3f}, pTM {ptm_mean:.3f}", flush=True)
