import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.protein_utils import prepare_protein_features

ESMFOLD2_REPO = "biohub/ESMFold2-Fast"
ESMFOLD_VARIANT = "fast"

# Keys infer_protein attaches in newer transformers; needed for output_to_pdb.
_PDB_FEATURE_KEYS = (
    "res_type",
    "atom_to_token",
    "ref_atom_name_chars",
    "atom_attention_mask",
    "token_attention_mask",
    "residue_index",
)


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


def _output_to_pdb(model: ESMFold2Model, infer_output: dict) -> str:
    """PDB export: model helper, then protein_utils (newer images), else clear error."""
    if hasattr(model, "output_to_pdb"):
        return model.output_to_pdb(infer_output)
    try:
        from transformers.models.esmfold2.protein_utils import output_to_pdb

        return output_to_pdb(infer_output)
    except ImportError as exc:
        raise RuntimeError(
            "PDB export is not available in this container's transformers build. "
            "Rebuild fast_esmfold.sif with a current Biohub transformers (see esmfold_cu.def)."
        ) from exc


def fold_one(
    model: ESMFold2Model,
    sequence: str,
    num_loops: int,
    num_sampling_steps: int,
) -> dict:
    """Same flow as infer_protein; re-attach featurization keys for PDB on older builds."""
    if hasattr(model, "infer_protein"):
        out = model.infer_protein(
            sequence,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
        )
        # Newer infer_protein already merges _PDB_FEATURE_KEYS; older does not.
        if "res_type" not in out:
            features = prepare_protein_features(sequence)
            features = {k: v.to(model.device) for k, v in features.items()}
            for k in _PDB_FEATURE_KEYS:
                if k in features:
                    out[k] = features[k]
        return out

    features = prepare_protein_features(sequence)
    features = {k: v.to(model.device) for k, v in features.items()}
    with torch.inference_mode():
        out = model(
            **features,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
        )
    for k in _PDB_FEATURE_KEYS:
        out[k] = features[k]
    return out


def save_outputs(
    output_dir: str,
    name: str,
    variant: str,
    model: ESMFold2Model,
    sequence: str,
    infer_output: dict,
    num_loops: int,
    num_sampling_steps: int,
) -> None:
    out_dir = Path(output_dir) / name / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        pdb_str = _output_to_pdb(model, infer_output)
    except RuntimeError:
        if not hasattr(model, "infer_protein_as_pdb"):
            raise
        pdb_str = model.infer_protein_as_pdb(
            sequence,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
        )
    (out_dir / "structure.pdb").write_text(pdb_str)

    plddt = infer_output["plddt"]
    if torch.is_tensor(plddt):
        if plddt.dim() > 1 and plddt.shape[0] == 1:
            plddt = plddt[0]
        plddt = plddt.detach().cpu().numpy()
    np.save(out_dir / "plddt.npy", plddt)

    ptm = infer_output.get("ptm")
    if torch.is_tensor(ptm):
        ptm = float(ptm.reshape(-1)[0]) if ptm.numel() == 1 else float(ptm.mean())
    torch.save({"name": name, "variant": variant, "ptm": ptm}, out_dir / "outputs.pt")


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
    args = parser.parse_args()
    hub_cache = _enforce_offline(args.cache)

    seq_by_name = parse_inputs(args.input_dir)
    model = load_model(hub_cache)

    for name, sequence in seq_by_name.items():
        print(f"Folding {name} ({len(sequence)} aa)...", flush=True)
        out = fold_one(model, sequence, args.num_loops, args.num_sampling_steps)
        save_outputs(
            args.output_dir,
            name,
            ESMFOLD_VARIANT,
            model,
            sequence,
            out,
            args.num_loops,
            args.num_sampling_steps,
        )
        plddt = out["plddt"]
        plddt_mean = float(plddt.mean()) if torch.is_tensor(plddt) else float(np.mean(plddt))
        ptm = out.get("ptm")
        ptm_mean = float(ptm.mean()) if torch.is_tensor(ptm) else float(ptm)
        print(f"  {name}: pLDDT mean {plddt_mean:.3f}, pTM {ptm_mean:.3f}", flush=True)
