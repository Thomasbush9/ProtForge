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


# bf16 tensor-core inference path (see run_batch_esmc.py). TF32 lifts fp32
# matmuls onto tensor cores; bf16 autocast halves matmul cost. SAE activations
# are detached to fp on CPU before saving, so storage dtype is unchanged.
# None on CPU-only hosts (autocast becomes a no-op).
def _enable_tf32() -> "torch.dtype | None":
    if not torch.cuda.is_available():
        return None
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.bfloat16


AUTOCAST_DTYPE: "torch.dtype | None" = None


# See run_batch_esmc.py — same length-bucketed micro-batcher (padded cost
# rows * max_len near budget). SAE forwards add the heavy SAE codebooks on top
# of the backbone, so the per-size budget is smaller than the embedding runner.
DEFAULT_TOKEN_BUDGET = {"6B": 6000, "600M": 24000, "300M": 32000}


def token_budget_batches(names, seqs, budget):
    """Yield lists of names grouped so each batch's padded cost (rows * max_len)
    stays near `budget`. Length-sorted; a single seq longer than budget forms
    its own batch (never dropped)."""
    order = sorted(range(len(names)), key=lambda i: len(seqs[i]))
    batch, max_len = [], 0
    for i in order:
        new_max = max(max_len, len(seqs[i]))
        if batch and (len(batch) + 1) * new_max > budget:
            yield [names[j] for j in batch]
            batch, max_len, new_max = [], 0, len(seqs[i])
        batch.append(i)
        max_len = new_max
    if batch:
        yield [names[j] for j in batch]


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


def forward_sae_batch(model, tokenizer, seqs):
    """Forward a length-bucketed micro-batch. Returns (sae_outputs, nonpad_counts)
    where sae_outputs[layer] is a sparse (sum_nonpad, n_features) tensor with rows
    in (batch, token) row-major order, and nonpad_counts[i] is sequence i's
    non-pad token count (BOS + residues + EOS). The caller re-splits by walking
    cumulative offsets and dropping the leading BOS / trailing EOS per segment
    (modeling_esmc_sae.py:109 flattens layer_states[token_mask] across batch+seq).
    """
    enc = tokenizer(seqs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in enc.items()}
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=AUTOCAST_DTYPE, enabled=AUTOCAST_DTYPE is not None
    ):
        output = model(**inputs)
    nonpad_counts = enc["attention_mask"].sum(dim=1).tolist()
    return output.get("sae_outputs", {}), nonpad_counts


def split_sae_outputs(sae_outputs: dict, nonpad_counts: list, seq_lens: list) -> list:
    """Re-split a batched, flattened sparse SAE output into per-sequence dense
    per-residue activations.

    sae_outputs[layer] is sparse (sum_nonpad, n_features) in (batch, token) order.
    For sequence i its block is rows [offset_i : offset_i + nonpad_i]; row 0 of
    the block is BOS and the last is EOS, so the real residues are the inner
    seq_len rows: [offset_i + 1 : offset_i + 1 + seq_len_i]. Fails loudly on any
    count mismatch — a silent misalignment would corrupt every saved activation.
    Returns a list (one dict {layer: dense (seq_len, n_features)} per sequence).
    """
    dense = {k: (v.to_dense() if v.is_sparse else v).detach().cpu()
             for k, v in sae_outputs.items() if torch.is_tensor(v)}
    total = sum(nonpad_counts)
    for k, t in dense.items():
        if t.shape[0] != total:
            raise SystemExit(
                f"SAE layer {k!r}: {t.shape[0]} flattened rows != summed non-pad "
                f"tokens {total}. Batch re-split would be misaligned; refusing to "
                "save corrupt activations.")
    per_seq = []
    offset = 0
    for nonpad, seq_len in zip(nonpad_counts, seq_lens):
        if nonpad != seq_len + 2:
            raise SystemExit(
                f"SAE re-split: non-pad tokens {nonpad} != seq_len+2 "
                f"({seq_len}+2). Expected exactly one BOS + one EOS per sequence; "
                "token layout changed — fix before trusting activations.")
        lo, hi = offset + 1, offset + 1 + seq_len   # drop BOS (row 0) and EOS
        per_seq.append({k: t[lo:hi] for k, t in dense.items()})
        offset += nonpad
    return per_seq


def save_sae(output_dir: str, name: str, size: str, sae_type: str,
             layer_acts: dict) -> int:
    """Save one sequence's per-layer per-residue activations under
    {output_dir}/{name}/{size}/{sae_type}/. layer_acts is {layer: (seq_len, D)}
    already re-split + densified by split_sae_outputs."""
    out_dir = Path(output_dir) / name / size / sae_type
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for key, tensor in layer_acts.items():
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
    # Batch mode: one process loads model+SAE ONCE and serves every chunk in
    # --input-dirs (parallel to --output-dirs) — group persistence. Debug mode
    # (--yaml / --sequence) prints per-residue shapes for one sequence.
    parser.add_argument("--input-dirs", nargs="+", default=None,
                        help="Chunk directories of YAMLs to encode (group)")
    parser.add_argument("--output-dirs", nargs="+", default=None,
                        help="Per-chunk output roots, parallel to --input-dirs")
    parser.add_argument("--token-budget", type=int, default=None,
                        help="Padded-token budget per micro-batch (rows * max_len). "
                             "Default: per-size DEFAULT_TOKEN_BUDGET.")
    parser.add_argument("--sequence", type=str, default=None, help="Raw sequence (single, debug)")
    parser.add_argument("--yaml", type=str, default=None, help="Single Boltz YAML (debug)")
    args = parser.parse_args()

    modes = [bool(args.input_dirs), bool(args.sequence), bool(args.yaml)]
    if sum(modes) != 1:
        parser.error("provide exactly one of --input-dirs, --sequence, --yaml")
    if args.input_dirs and not args.output_dirs:
        parser.error("--input-dirs requires --output-dirs")
    if args.input_dirs and len(args.input_dirs) != len(args.output_dirs):
        parser.error("--input-dirs and --output-dirs must be equal length")

    hub_cache = _enforce_offline(args.cache)
    model_repo = resolve_model_repo(args.size, args.model)
    sae_repo = resolve_sae_repo(args.size, args.sae_type, args.sae_repo)
    layer_ids = parse_layers(args.layers, args.size, hub_cache, sae_repo)

    print(f"size={args.size} model={model_repo} sae={sae_repo} "
          f"layers={layer_ids} cache={hub_cache}", flush=True)

    AUTOCAST_DTYPE = _enable_tf32()

    model, tokenizer = load_model_with_sae(
        model_repo=model_repo, sae_repo=sae_repo,
        hub_cache=hub_cache, layer_ids=layer_ids,
    )

    if args.input_dirs:
        budget = args.token_budget or DEFAULT_TOKEN_BUDGET.get(args.size, 12000)
        # Flatten the group, tagging each seq by chunk index so identical stems
        # in different chunks never collide; carry each chunk's output dir.
        tags, seqs, out_by_tag, name_by_tag, len_by_tag = [], [], {}, {}, {}
        for ci, in_dir in enumerate(args.input_dirs):
            for name, sequence in parse_inputs(in_dir).items():
                tag = f"{ci}/{name}"
                tags.append(tag); seqs.append(sequence)
                out_by_tag[tag] = args.output_dirs[ci]
                name_by_tag[tag] = name; len_by_tag[tag] = len(sequence)
        seq_by_tag = dict(zip(tags, seqs))
        for batch_tags in token_budget_batches(tags, seqs, budget):
            batch_seqs = [seq_by_tag[t] for t in batch_tags]
            seq_lens = [len_by_tag[t] for t in batch_tags]
            print(f"Extracting SAE for {len(batch_tags)} seq(s) "
                  f"(max {max(seq_lens)} aa)...", flush=True)
            sae_outputs, nonpad = forward_sae_batch(model, tokenizer, batch_seqs)
            if not sae_outputs:
                raise RuntimeError(f"No sae_outputs for batch {batch_tags}")
            per_seq = split_sae_outputs(sae_outputs, nonpad, seq_lens)
            for tag, layer_acts in zip(batch_tags, per_seq):
                n = save_sae(out_by_tag[tag], name_by_tag[tag], args.size,
                             args.sae_type, layer_acts)
            print(f"  saved {n} layer file(s) x {len(batch_tags)} seq(s)", flush=True)
    else:
        sequence = (args.sequence or sequence_from_yaml(args.yaml)).strip()
        sae_outputs, nonpad = forward_sae_batch(model, tokenizer, [sequence])
        if not sae_outputs:
            raise RuntimeError("No sae_outputs in model forward pass")
        per_seq = split_sae_outputs(sae_outputs, nonpad, [len(sequence)])
        for key, tensor in sorted(per_seq[0].items()):
            print(f"sae_outputs[{key!r}] per-residue shape = "
                  f"{tuple(tensor.shape)}", flush=True)
