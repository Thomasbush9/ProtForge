import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml
from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    ProteinInput,
    StructurePredictionInput,
)
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


def save_outputs(output_dir: str, name: str, variant: str, result) -> None:
    out_dir = Path(output_dir) / name / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "structure.cif").write_text(result.complex.to_mmcif())

    plddt = result.plddt
    if torch.is_tensor(plddt):
        plddt = plddt.detach().cpu().numpy()
    np.save(out_dir / "plddt.npy", np.asarray(plddt))

    meta = {"name": name, "variant": variant, "ptm": result.ptm}
    if result.iptm is not None:
        meta["iptm"] = result.iptm
    torch.save(meta, out_dir / "metrics.pt")


def load_model(hub_cache: str) -> ESMFold2Model:
    return ESMFold2Model.from_pretrained(
        ESMFOLD2_REPO,
        cache_dir=hub_cache,
        local_files_only=True,
    ).cuda().eval()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-loops", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    hub_cache = _enforce_offline(args.cache)

    seq_by_name = parse_inputs(args.input_dir)
    model = load_model(hub_cache)
    builder = ESMFold2InputBuilder()

    for name, sequence in seq_by_name.items():
        print(f"Folding {name} ({len(sequence)} aa)...", flush=True)
        spi = StructurePredictionInput(
            sequences=[ProteinInput(id="A", sequence=sequence)]
        )
        result = builder.fold(
            model,
            spi,
            num_loops=args.num_loops,
            num_sampling_steps=args.num_sampling_steps,
            num_diffusion_samples=1,
            seed=args.seed,
        )
        save_outputs(args.output_dir, name, ESMFOLD_VARIANT, result)

        plddt_mean = float(result.plddt.mean())
        ptm = float(result.ptm)
        msg = f"  {name}: pLDDT mean {plddt_mean:.3f}, pTM {ptm:.3f}"
        if result.iptm is not None:
            msg += f", ipTM {float(result.iptm):.3f}"
        print(msg, flush=True)
