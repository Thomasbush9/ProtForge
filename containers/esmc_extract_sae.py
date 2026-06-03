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
    "600M": [18, 36],
    "300M": [12, 24],
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


def discover_all_layers(hub_cache: str | None, sae_repo: str) -> list[int]:
    """Enumerate every layer_<id>.safetensors shard cached for an SAE repo.

    Used for --layers all: the all-layers SAE repo ships one shard per trained
    layer, so the layer set is whatever is present in the local snapshot.
    """
    if not hub_cache:
        raise ValueError("--layers all requires a populated --cache to enumerate SAE shards")
    repo_dir = Path(hub_cache) / f"models--{sae_repo.replace('/', '--')}" / "snapshots"
    ids: set[int] = set()
    for shard in repo_dir.glob("*/layer_*.safetensors"):
        try:
            ids.add(int(shard.stem[len("layer_"):]))
        except ValueError:
            continue
    if not ids:
        raise FileNotFoundError(
            f"No layer_*.safetensors under {repo_dir}; cannot resolve --layers all"
        )
    return sorted(ids)


def parse_layers(layers: str | None, size: str, hub_cache: str | None, sae_repo: str) -> list[int]:
    if layers and layers.strip().lower() == "all":
        return discover_all_layers(hub_cache, sae_repo)
    if layers:
        return [int(x.strip()) for x in layers.split(",") if x.strip()]
    if size in DEFAULT_LAYERS:
        return DEFAULT_LAYERS[size]
    raise ValueError(
        f"No default SAE layers for --size {size!r}; pass --layers (comma-separated, or 'all')"
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


def parse_inputs(input_dir: str) -> dict[str, str]:
    """Map YAML stem -> sequence for every *.yaml in a chunk directory."""
    paths = sorted(Path(input_dir).glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No *.yaml in {input_dir}")
    out: dict[str, str] = {}
    for p in paths:
        stem = p.stem
        if stem in out:
            raise ValueError(f"Duplicate stem {stem!r} in {input_dir}")
        out[stem] = sequence_from_yaml(str(p))
    return out


def sae_allow_patterns(layer_ids: list[int]) -> list[str]:
    patterns = ["config.json"]
    patterns.extend(f"layer_{layer_id}.safetensors" for layer_id in layer_ids)
    return patterns


def load_model_with_sae(
    *,
    model_repo: str,
    sae_repo: str,
    hub_cache: str | None,
    layer_ids: list[int],
):
    """Load the ESMC model + tokenizer and attach the requested SAE layers."""
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
    return model, tokenizer


def forward_sae(model, tokenizer, sequence: str) -> dict:
    """Run one sequence through the model and return its SAE activations."""
    inputs = tokenizer(sequence, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs)
    return output.get("sae_outputs", {})


def save_sae(
    output_dir: str,
    name: str,
    size: str,
    sae_type: str,
    sae_outputs: dict,
    seq_len: int,
) -> int:
    """Save per-layer SAE activations under {output_dir}/{name}/{size}/{sae_type}/.

    One forward per sequence, so the batch dim is 1 — drop it and trim to the
    real sequence length, mirroring run_batch_esmc.py's save_outputs.
    """
    out_dir = Path(output_dir) / name / size / sae_type
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for key, tensor in sae_outputs.items():
        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu()
            if tensor.dim() >= 2 and tensor.shape[0] == 1:
                tensor = tensor[0, :seq_len]
        safe = str(key).replace("/", "_")
        torch.save(
            {"activations": tensor, "layer": key, "name": name,
             "size": size, "sae_type": sae_type},
            out_dir / f"sae_{safe}.pt",
        )
        saved += 1
    return saved


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract ESMC SAE sparse activations.")
    parser.add_argument("--cache", type=str, required=True,
                        help="HF_HOME root (bind-mounted cache, e.g. /models/hf)")
    parser.add_argument("--size", type=str, default="6B", choices=list(ESMC_MODELS),
                        help="ESMC checkpoint size; selects model + SAE repo unless overridden")
    parser.add_argument("--sae", dest="sae_type", type=str, default="all-layers",
                        choices=list(SAE_REPOS), help="SAE variant (repo id derived from --size)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override HF model repo (default: biohub/ESMC-<size>)")
    parser.add_argument("--sae-repo", type=str, default=None,
                        help="Override HF SAE repo (default: derived from --size and --sae)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated SAE layer indices, or 'all' (default: per-size defaults)")
    # Batch mode (directory of YAMLs -> saved activations) vs single-sequence
    # debug mode (--yaml / --sequence, prints shapes). Exactly one is required.
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory of Boltz YAMLs to encode in batch")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output root for batch mode (required with --input-dir)")
    parser.add_argument("--sequence", type=str, default=None, help="Raw sequence (single, debug)")
    parser.add_argument("--yaml", type=str, default=None, help="Single Boltz YAML (debug)")
    args = parser.parse_args()

    modes = [bool(args.input_dir), bool(args.sequence), bool(args.yaml)]
    if sum(modes) != 1:
        parser.error("provide exactly one of --input-dir, --sequence, --yaml")
    if args.input_dir and not args.output_dir:
        parser.error("--input-dir requires --output-dir")

    hub_cache = _enforce_offline(args.cache)
    model_repo = resolve_model_repo(args.size, args.model)
    sae_repo = resolve_sae_repo(args.size, args.sae_type, args.sae_repo)
    layer_ids = parse_layers(args.layers, args.size, hub_cache, sae_repo)

    print(f"size={args.size} model={model_repo} sae={sae_repo} "
          f"layers={layer_ids} cache={hub_cache}", flush=True)

    model, tokenizer = load_model_with_sae(
        model_repo=model_repo, sae_repo=sae_repo,
        hub_cache=hub_cache, layer_ids=layer_ids,
    )

    if args.input_dir:
        seq_by_name = parse_inputs(args.input_dir)
        for name, sequence in seq_by_name.items():
            print(f"Extracting SAE for {name} ({len(sequence)} aa)...", flush=True)
            sae_outputs = forward_sae(model, tokenizer, sequence)
            if not sae_outputs:
                raise RuntimeError(f"No sae_outputs for {name}")
            n = save_sae(args.output_dir, name, args.size, args.sae_type,
                         sae_outputs, len(sequence))
            print(f"  saved {n} layer file(s)", flush=True)
    else:
        sequence = (args.sequence or sequence_from_yaml(args.yaml)).strip()
        sae_outputs = forward_sae(model, tokenizer, sequence)
        if not sae_outputs:
            raise RuntimeError("No sae_outputs in model forward pass")
        for key, tensor in sorted(sae_outputs.items()):
            print(f"sae_outputs[{key!r}].shape = {tuple(tensor.shape)}", flush=True)
