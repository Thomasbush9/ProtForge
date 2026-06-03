import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import torch
import yaml
from transformers import AutoTokenizer
from transformers.models.esmc.modeling_esmc import ESMCModel

ESMC_MODELS = {
    "6B": "biohub/ESMC-6B",
    "600M": "biohub/ESMC-600M",
    "300M": "biohub/ESMC-300M",
}


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


def save_outputs(output_dir: str, name: str, size: str, model_output, idx: int, seq_len: int) -> None:
    out_dir = Path(output_dir) / name / size
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden = model_output.last_hidden_state[idx, :seq_len].cpu()
    torch.save(
        {"last_hidden_state": hidden, "name": name, "size": size},
        out_dir / "outputs.pt",
    )


def encode_sequences(sequences: List, model_name, hub_cache):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=hub_cache,
        local_files_only=True,
    )
    model = ESMCModel.from_pretrained(
        model_name,
        cache_dir=hub_cache,
        local_files_only=True,
    ).cuda().eval()
    inputs = tokenizer(sequences, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # Time only the forward pass (CUDA-synced) for the stair benchmark's
    # infer_s column; model load / tokenization are excluded by construction.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"BENCH_INFER_S {model_name} {time.perf_counter() - _t0:.4f}", flush=True)
    return outputs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--size", type=str, default="6B", choices=[*ESMC_MODELS, "all"])
    args = parser.parse_args()
    hub_cache = _enforce_offline(args.cache)

    seq_by_name = parse_inputs(args.input_dir)
    names = list(seq_by_name.keys())
    sequences = [seq_by_name[n] for n in names]
    lengths = {n: len(seq_by_name[n]) for n in names}

    if args.size == "all":
        for size, model_name in ESMC_MODELS.items():
            out = encode_sequences(sequences, model_name, hub_cache)
            for i, name in enumerate(names):
                save_outputs(args.output_dir, name, size, out, i, lengths[name])
    else:
        out = encode_sequences(sequences, ESMC_MODELS[args.size], hub_cache)
        for i, name in enumerate(names):
            save_outputs(args.output_dir, name, args.size, out, i, lengths[name])
