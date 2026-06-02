import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.protein_utils import (
    OUTPUT_TO_PDB_FEATURE_KEYS,
    output_to_pdb,
    prepare_protein_features,
)

ESMFOLD2_REPO = "biohub/ESMFold2-Fast"
ESMFOLD_VARIANT = "fast"

_TOKEN_KEYS = (
    "token_index",
    "residue_index",
    "asym_id",
    "sym_id",
    "entity_id",
    "mol_type",
    "res_type",
    "input_ids",
    "distogram_atom_idx",
    "deletion_mean",
)
_ATOM_KEYS = ("ref_element", "ref_charge", "ref_space_uid", "atom_to_token")
_MSA_KEYS = ("msa", "msa_attention_mask", "has_deletion", "deletion_value")


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


def collate_protein_features(per_sample: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad and stack single-protein feature dicts (B=1 each) to batch size B."""
    items = [{k: v.squeeze(0) for k, v in f.items()} for f in per_sample]
    max_L = max(x["res_type"].shape[0] for x in items)
    max_atoms = max(x["ref_pos"].shape[0] for x in items)

    out: dict[str, Tensor] = {}
    for key in _TOKEN_KEYS:
        out[key] = pad_sequence(
            [x[key] for x in items], batch_first=True, padding_value=0
        )

    out["token_attention_mask"] = pad_sequence(
        [x["token_attention_mask"] for x in items],
        batch_first=True,
        padding_value=False,
    )

    bonds = []
    for x in items:
        L = x["token_bonds"].shape[0]
        t = x["token_bonds"]
        pad = torch.zeros(max_L, max_L, t.shape[-1], dtype=t.dtype)
        pad[:L, :L] = t
        bonds.append(pad)
    out["token_bonds"] = torch.stack(bonds, dim=0)

    ref_pos = []
    for x in items:
        n = x["ref_pos"].shape[0]
        p = torch.zeros(max_atoms, 3, dtype=x["ref_pos"].dtype)
        p[:n] = x["ref_pos"]
        ref_pos.append(p)
    out["ref_pos"] = torch.stack(ref_pos, dim=0)

    for key in _ATOM_KEYS:
        out[key] = pad_sequence(
            [x[key] for x in items], batch_first=True, padding_value=0
        )
    out["ref_atom_name_chars"] = pad_sequence(
        [x["ref_atom_name_chars"] for x in items],
        batch_first=True,
        padding_value=0,
    )
    out["atom_attention_mask"] = pad_sequence(
        [x["atom_attention_mask"] for x in items],
        batch_first=True,
        padding_value=False,
    )

    for key in _MSA_KEYS:
        pad_val = False if key in ("msa_attention_mask", "has_deletion") else 0
        padded = pad_sequence(
            [x[key].squeeze(0) for x in items],
            batch_first=True,
            padding_value=pad_val,
        )
        out[key] = padded.unsqueeze(1)

    return out


def _slice_batch_output(
    batch_output: dict, batch_features: dict[str, Tensor], idx: int
) -> dict:
    """One sample with leading batch dim 1 (for output_to_pdb)."""
    single: dict = {}
    for k, v in batch_output.items():
        if isinstance(v, Tensor) and v.ndim >= 1 and v.shape[0] > idx:
            single[k] = v[idx : idx + 1]
        else:
            single[k] = v
    for k in OUTPUT_TO_PDB_FEATURE_KEYS:
        if k in batch_features:
            single[k] = batch_features[k][idx : idx + 1]
    return single


def _infer_output_to_pdb(_model: ESMFold2Model, infer_output) -> str:
    if hasattr(_model, "output_to_pdb"):
        return _model.output_to_pdb(infer_output)
    return output_to_pdb(infer_output)


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
        if plddt.dim() > 1 and plddt.shape[0] == 1:
            plddt = plddt[0]
        plddt = plddt.detach().cpu().numpy()
    np.save(out_dir / "plddt.npy", plddt)

    ptm = infer_output.get("ptm")
    if torch.is_tensor(ptm):
        ptm = float(ptm.reshape(-1)[0]) if ptm.numel() == 1 else float(ptm.mean())
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


def fold_sequences_batch(
    model: ESMFold2Model,
    sequences: list[str],
    num_loops: int,
    num_sampling_steps: int,
) -> tuple[dict, dict[str, Tensor]]:
    """One GPU forward for B independent monomers (padded batch)."""
    features_list = [prepare_protein_features(seq) for seq in sequences]
    features = collate_protein_features(features_list)
    features = {k: v.to(model.device) for k, v in features.items()}
    with torch.inference_mode():
        output = model(
            **features,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
        )
    for k in OUTPUT_TO_PDB_FEATURE_KEYS:
        output[k] = features[k]
    return output, features


def _log_metrics(name: str, infer_output: dict) -> None:
    plddt = infer_output["plddt"]
    plddt_mean = float(plddt.mean()) if torch.is_tensor(plddt) else float(np.mean(plddt))
    ptm = infer_output.get("ptm")
    ptm_mean = float(ptm.mean()) if torch.is_tensor(ptm) else float(ptm)
    print(f"  {name}: pLDDT mean {plddt_mean:.3f}, pTM {ptm_mean:.3f}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-loops", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Sequences per GPU forward (0 = all inputs in one batch)",
    )
    args = parser.parse_args()
    hub_cache = _enforce_offline(args.cache)

    seq_by_name = parse_inputs(args.input_dir)
    names = list(seq_by_name.keys())
    sequences = [seq_by_name[n] for n in names]
    batch_size = args.batch_size or len(sequences)

    model = load_model(hub_cache)

    for start in range(0, len(names), batch_size):
        batch_names = names[start : start + batch_size]
        batch_seqs = sequences[start : start + batch_size]
        lens = ", ".join(f"{n}={len(s)}" for n, s in zip(batch_names, batch_seqs))
        print(f"Folding batch [{lens}]...", flush=True)

        batch_out, batch_feats = fold_sequences_batch(
            model,
            batch_seqs,
            args.num_loops,
            args.num_sampling_steps,
        )
        for i, name in enumerate(batch_names):
            single = _slice_batch_output(batch_out, batch_feats, i)
            save_outputs(args.output_dir, name, ESMFOLD_VARIANT, model, single)
            _log_metrics(name, single)
