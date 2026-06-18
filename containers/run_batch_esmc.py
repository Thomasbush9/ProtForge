import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer
from transformers.models.esmc.modeling_esmc import ESMCModel

ESMC_MODELS = {
    "6B": "biohub/ESMC-6B",
    "600M": "biohub/ESMC-600M",
    "300M": "biohub/ESMC-300M",
}

# The 20 canonical amino acids — saved logit columns for these tokens are all a
# mutation scan needs (see workflow/scripts/mutation_scan.py).
CANONICAL_AAS = "ACDEFGHIKLMNPQRSTVWY"


def _enforce_offline(cache_dir: str | None) -> str | None:
    if not cache_dir:
        return None
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return f"{cache_dir}/hub"


# bf16 tensor-core inference path. TF32 lifts fp32 matmuls onto tensor cores
# (free on Ampere+); bf16 autocast halves the matmul cost again with a wide
# enough exponent that embeddings/logits stay within fp tolerance. Logit
# outputs are cast back to fp32 on CPU before saving, so downstream LLR math is
# unchanged. AUTOCAST_DTYPE is None on CPU-only hosts (autocast becomes a no-op).
def _enable_tf32() -> "torch.dtype | None":
    if not torch.cuda.is_available():
        return None
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.bfloat16


AUTOCAST_DTYPE: "torch.dtype | None" = None


def _seq_from_yaml(path: Path) -> str:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data or "sequences" not in data:
        raise KeyError(f"{path}: missing 'sequences'")
    entry = data["sequences"][0]
    if "protein" not in entry or "sequence" not in entry["protein"]:
        raise KeyError(f"{path}: expected sequences[0].protein.sequence")
    return str(entry["protein"]["sequence"]).strip()


def _seq_from_fasta(path: Path) -> str:
    """Return the sequence of the single record in a FASTA file."""
    records, cur = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                records.append("".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        records.append("".join(cur))
    if not records:
        raise ValueError(f"{path}: no sequence found")
    if len(records) > 1:
        raise ValueError(f"{path}: expected one sequence per FASTA, found {len(records)}")
    return records[0]


def parse_inputs(input_dir: str) -> dict[str, str]:
    """Read sequences keyed by file stem from .yaml and/or .fasta/.fa inputs."""
    base = Path(input_dir)
    paths = sorted([*base.glob("*.yaml"), *base.glob("*.fasta"), *base.glob("*.fa")])
    if not paths:
        raise FileNotFoundError(f"No *.yaml/*.fasta in {input_dir}")

    out: dict[str, str] = {}
    for p in paths:
        seq = _seq_from_yaml(p) if p.suffix == ".yaml" else _seq_from_fasta(p)
        stem = p.stem
        if stem in out:
            raise ValueError(f"Duplicate stem {stem!r} in {input_dir}")
        out[stem] = seq
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
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=AUTOCAST_DTYPE, enabled=AUTOCAST_DTYPE is not None
    ):
        outputs = model(**inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"BENCH_INFER_S {model_name} {time.perf_counter() - _t0:.4f}", flush=True)
    return outputs


def encode_for_logits(sequences: List, model_name, hub_cache):
    """Forward pass with the LM-head model to get per-residue vocab logits.

    The base ESMCModel has no LM head and cannot produce logits, so we load
    AutoModelForMaskedLM (the released biohub/ESMC-* checkpoints ship the head;
    EvolutionaryScale document it for zero-shot variant scoring). We request a
    special-tokens mask so real residue positions can be selected exactly —
    never assume a fixed BOS/EOS offset.

    Returns (logits, special_tokens_mask, attention_mask, aa_token_ids) with
    logits as a CPU float32 numpy array of shape (batch, tokens, vocab).
    """
    from transformers import AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=hub_cache, local_files_only=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, cache_dir=hub_cache, local_files_only=True,
    ).cuda().eval()
    enc = tokenizer(
        sequences, return_tensors="pt", padding=True,
        return_special_tokens_mask=True,
    )
    inputs = {k: v.to(model.device)
              for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=AUTOCAST_DTYPE, enabled=AUTOCAST_DTYPE is not None
    ):
        outputs = model(**inputs)
    aa_token_ids = {
        aa: int(tid) for aa, tid in
        zip(CANONICAL_AAS, tokenizer.convert_tokens_to_ids(list(CANONICAL_AAS)))
    }
    return (
        outputs.logits.float().cpu().numpy(),
        enc["special_tokens_mask"].cpu().numpy(),
        enc["attention_mask"].cpu().numpy(),
        aa_token_ids,
    )


def save_logits_outputs(output_dir: str, name: str, size: str,
                        logits_real: "np.ndarray", aa_token_ids: dict) -> None:
    """Write per-residue logits (fp16) + the AA->token-id map for one sequence.

    logits_real is (L_real, vocab) for real residues only. Downstream scoring
    (workflow/scripts/mutation_scan.py) reads these with NumPy — no torch.
    """
    out_dir = Path(output_dir) / name / size
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "logits.npy", logits_real.astype(np.float16))
    (out_dir / "aa_token_ids.json").write_text(json.dumps(aa_token_ids))


def _real_residue_logits(logits, special_mask, attn_mask, idx, seq_len, name):
    """Select the real-residue rows for sequence idx and verify the count.

    Fails loudly if the number of non-special, non-padding positions doesn't
    equal the sequence length — that mismatch means logit rows are misaligned
    with residues and any LLR derived from them would be silently wrong.
    """
    real = (special_mask[idx] == 0) & (attn_mask[idx] == 1)
    rows = logits[idx][real]
    if rows.shape[0] != seq_len:
        raise SystemExit(
            f"{name}: {rows.shape[0]} real-residue logit rows != sequence length "
            f"{seq_len}. Token/residue alignment is off — fix before trusting "
            "any mutation scan derived from these logits."
        )
    return rows


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--size", type=str, default="6B", choices=[*ESMC_MODELS, "all"])
    parser.add_argument(
        "--save-logits", action="store_true",
        help="Produce per-residue vocabulary logits (logits.npy + "
             "aa_token_ids.json) for zero-shot mutation scanning instead of "
             "embeddings. Loads the LM-head model (AutoModelForMaskedLM).",
    )
    args = parser.parse_args()
    hub_cache = _enforce_offline(args.cache)
    AUTOCAST_DTYPE = _enable_tf32()

    seq_by_name = parse_inputs(args.input_dir)
    names = list(seq_by_name.keys())
    sequences = [seq_by_name[n] for n in names]
    lengths = {n: len(seq_by_name[n]) for n in names}

    sizes = list(ESMC_MODELS) if args.size == "all" else [args.size]

    for size in sizes:
        model_name = ESMC_MODELS[size]
        if args.save_logits:
            logits, special_mask, attn_mask, aa_token_ids = encode_for_logits(
                sequences, model_name, hub_cache)
            for i, name in enumerate(names):
                rows = _real_residue_logits(
                    logits, special_mask, attn_mask, i, lengths[name], name)
                save_logits_outputs(args.output_dir, name, size, rows, aa_token_ids)
        else:
            out = encode_sequences(sequences, model_name, hub_cache)
            for i, name in enumerate(names):
                save_outputs(args.output_dir, name, size, out, i, lengths[name])
