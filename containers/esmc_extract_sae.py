import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml
from transformers import AutoModel, AutoTokenizer
from transformers.models.esmc.modeling_esmc import ESMCModel

ESMC_MODELS = {
    "6B": "biohub/ESMC-6B",
    "600M": "biohub/ESMC-600M",
    "300M": "biohub/ESMC-300M",
}

SAE_REPOS = {
    "all-layers": "biohub/ESMC-{size}-sae-k64-codebook16384",
    "mlp": "biohub/ESMC-{size}-sae-mlp-k64-codebook131072",
}

DEFAULT_LAYERS = {
    "6B": [30, 60],
}


def _enforce_offline(cache_dir: str | None) -> str | None:
    if not cache_dir:
        return None
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return f"{cache_dir}/hub"


def resolve_model_repo(size: str, override: str | None) -> str:
    if override:
        return override
    try:
        return ESMC_MODELS[size]
    except KeyError as exc:
        raise ValueError(f"Unknown --size {size!r}; choose from {list(ESMC_MODELS)}") from exc


def resolve_sae_repo(size: str, sae_type: str, override: str | None) -> str:
    if override:
        return override
    try:
        return SAE_REPOS[sae_type].format(size=size)
    except KeyError as exc:
        raise ValueError(f"Unknown --sae {sae_type!r}; choose from {list(SAE_REPOS)}") from exc


def parse_layers(layers: str | None, size: str) -> list[int]:
    if layers:
        return [int(x.strip()) for x in layers.split(",") if x.strip()]
    if size in DEFAULT_LAYERS:
        return DEFAULT_LAYERS[size]
    raise ValueError(
        f"No default SAE layers for --size {size!r}; pass --layers (comma-separated layer indices)"
    )


def sequence_from_yaml(yaml_path: str) -> str:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    if not data or "sequences" not in data:
        raise KeyError(f"{yaml_path}: missing 'sequences'")
    entry = data["sequences"][0]
    if "protein" not in entry or "sequence" not in entry["protein"]:
        raise KeyError(f"{yaml_path}: expected sequences[0].protein.sequence")
    return str(entry["protein"]["sequence"]).strip()


def sae_allow_patterns(layer_ids: list[int]) -> list[str]:
    patterns = ["config.json"]
    patterns.extend(f"layer_{layer_id}.safetensors" for layer_id in layer_ids)
    return patterns


def load_sequence(args) -> str:
    if args.sequence:
        return args.sequence.strip()
    return sequence_from_yaml(args.yaml)


def extract_sae(
    *,
    model_repo: str,
    sae_repo: str,
    sequence: str,
    hub_cache: str | None,
    layer_ids: list[int],
) -> dict:
    print(f"Loading {model_repo}...", flush=True)
    model = ESMCModel.from_pretrained(
        model_repo,
        cache_dir=hub_cache,
        local_files_only=bool(hub_cache),
    ).cuda().eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_repo,
        cache_dir=hub_cache,
        local_files_only=bool(hub_cache),
    )

    print(f"Loading {sae_repo} (layers {layer_ids})...", flush=True)
    sae = AutoModel.from_pretrained(
        sae_repo,
        allow_patterns=sae_allow_patterns(layer_ids),
        cache_dir=hub_cache,
        local_files_only=bool(hub_cache),
        device=model.device,
    )
    sae.initialize_layers(layer_ids)
    model.add_sae_models([sae.layers[str(layer_id)] for layer_id in layer_ids])

    inputs = tokenizer(sequence, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model(**inputs)

    return output.get("sae_outputs", {})


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract ESMC SAE sparse activations for one sequence.")
    parser.add_argument(
        "--cache",
        type=str,
        required=True,
        help="HF_HOME root (bind-mounted cache, e.g. /models/hf)",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="6B",
        choices=list(ESMC_MODELS),
        help="ESMC checkpoint size; selects model + SAE repo unless overridden",
    )
    parser.add_argument(
        "--sae",
        dest="sae_type",
        type=str,
        default="all-layers",
        choices=list(SAE_REPOS),
        help="SAE variant (repo id derived from --size)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override HF model repo (default: biohub/ESMC-<size>)",
    )
    parser.add_argument(
        "--sae-repo",
        type=str,
        default=None,
        help="Override HF SAE repo (default: derived from --size and --sae)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated SAE layer indices (default: 30,60 for 6B all-layers)",
    )
    seq = parser.add_mutually_exclusive_group(required=True)
    seq.add_argument("--sequence", type=str, help="Raw amino-acid sequence")
    seq.add_argument(
        "--yaml",
        type=str,
        help="Boltz-style YAML; uses sequences[0].protein.sequence",
    )
    args = parser.parse_args()

    hub_cache = _enforce_offline(args.cache)
    model_repo = resolve_model_repo(args.size, args.model)
    sae_repo = resolve_sae_repo(args.size, args.sae_type, args.sae_repo)
    layer_ids = parse_layers(args.layers, args.size)
    sequence = load_sequence(args)

    print(
        f"size={args.size} model={model_repo} sae={sae_repo} "
        f"layers={layer_ids} len={len(sequence)} cache={hub_cache}",
        flush=True,
    )

    sae_outputs = extract_sae(
        model_repo=model_repo,
        sae_repo=sae_repo,
        sequence=sequence,
        hub_cache=hub_cache,
        layer_ids=layer_ids,
    )

    if not sae_outputs:
        raise RuntimeError("No sae_outputs in model forward pass")

    for key, tensor in sorted(sae_outputs.items()):
        print(f"sae_outputs[{key!r}].shape = {tuple(tensor.shape)}", flush=True)
