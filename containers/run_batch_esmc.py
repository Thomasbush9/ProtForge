import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path

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
# Residues that can appear in a wild type but are not substitution *targets*.
# Selenocysteine is a genuine residue in selenoproteins (the deiodinases, the
# glutathione peroxidases) and has its own ESM-C token, so scoring a Sec-bearing
# wild type needs its logit column for the LLR denominator. It is not a scan
# target because Sec insertion requires a SECIS element in the mRNA.
WT_ONLY_AAS = "U"


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


# Per-model padded-token budget for one micro-batch. ESM-C-6B is far heavier per
# token than 300M, so its budget is smaller to bound peak activation memory.
# budget ~= rows * padded_len; tune against the per-size slurm mem_mb if needed.
DEFAULT_TOKEN_BUDGET = {"6B": 8000, "600M": 32000, "300M": 49152}


def token_budget_batches(names, seqs, budget, pad_free=False):
    """Yield lists of names grouped to fit `budget` tokens, length-sorted.

    Cost model depends on the attention backend:
      * pad_free=False (sdpa/eager): padded cost = rows * max_len, since padded
        positions are still computed. Length-sorting keeps the pad tight.
      * pad_free=True (FlashAttention-2): cost = sum(lengths). FA2 unpads the
        batch (varlen-flat) so padded positions cost ZERO FLOPs/memory; sizing
        by summed real tokens packs more sequences per batch than the padded
        model would. Still length-sorted (keeps each batch's varlen segments
        similar, which helps kernel efficiency).
    A single sequence longer than `budget` forms its own batch (never dropped).
    """
    order = sorted(range(len(names)), key=lambda i: len(seqs[i]))
    batch, max_len, total = [], 0, 0
    for i in order:
        L = len(seqs[i])
        new_max = max(max_len, L)
        cost = (total + L) if pad_free else (len(batch) + 1) * new_max
        if batch and cost > budget:
            yield [names[j] for j in batch]
            batch, max_len, total, new_max = [], 0, 0, L
        batch.append(i)
        max_len, total = new_max, total + L
    if batch:
        yield [names[j] for j in batch]


def _attn_impl():
    """Prefer FlashAttention-2 for ESM-C: it unpads the batch into a varlen-flat
    layout (modeling_esmc.py unpad_input/pad_input), so PADDED positions cost
    ZERO FLOPs — the real fix for batch padding, beyond length bucketing which
    only shrinks the pad. sdpa/eager mask padding but still COMPUTE it. FA2
    needs CUDA + the flash_attn package (both present in the ESM SIF) and bf16
    activations (our autocast provides them). Single-chain inputs only, which is
    always our case (one protein per file). Falls back to the HF default when
    flash_attn is unavailable (e.g. CPU host) so the runner still imports.
    """
    if not torch.cuda.is_available():
        return None
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return None
    return "flash_attention_2"


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


def save_outputs(output_dir: str, name: str, size: str, hidden: "torch.Tensor") -> None:
    """Persist one sequence's per-residue embeddings (already sliced to its real
    length and moved to CPU by the batched caller)."""
    out_dir = Path(output_dir) / name / size
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"last_hidden_state": hidden, "name": name, "size": size},
        out_dir / "outputs.pt",
    )


def load_embed_model(model_name, hub_cache):
    """Load the ESM-C embedding model + tokenizer ONCE; reused across every
    micro-batch of every chunk in the group (load-once-serve-many)."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=hub_cache, local_files_only=True,
    )
    model = ESMCModel.from_pretrained(
        model_name, cache_dir=hub_cache, local_files_only=True,
        attn_implementation=_attn_impl(),
    ).cuda().eval()
    return model, tokenizer


def embed_batch(model, tokenizer, seqs, label):
    """Forward one length-bucketed micro-batch. Returns last_hidden_state on the
    device; the caller slices each row to its real length. Padded positions are
    masked inside ESMCModel (FA2 unpad / sdpa sequence_id), so per-row hidden
    states are uncontaminated by neighbours in the batch."""
    inputs = tokenizer(seqs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=AUTOCAST_DTYPE, enabled=AUTOCAST_DTYPE is not None
    ):
        outputs = model(**inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"BENCH_INFER_S {label} rows={len(seqs)} "
          f"{time.perf_counter() - _t0:.4f}", flush=True)
    return outputs.last_hidden_state


def build_aa_token_ids(tokenizer):
    """Map each scorable residue letter to its vocabulary column.

    Covers the canonical 20 plus WT_ONLY_AAS. A canonical residue the vocabulary
    cannot represent is fatal — the LLR math would silently read the unknown
    token's logit. An optional residue is simply left out, so a tokenizer
    without a Sec token still yields a usable map for ordinary proteins.
    """
    wanted = CANONICAL_AAS + WT_ONLY_AAS
    ids = tokenizer.convert_tokens_to_ids(list(wanted))
    unk = getattr(tokenizer, "unk_token_id", None)

    aa_token_ids = {}
    for aa, tid in zip(wanted, ids):
        if tid is None or (unk is not None and tid == unk):
            if aa in CANONICAL_AAS:
                raise RuntimeError(
                    f"tokenizer has no token for canonical residue {aa!r} — "
                    "cannot score mutations against this vocabulary."
                )
            continue
        aa_token_ids[aa] = int(tid)
    return aa_token_ids


def load_logits_model(model_name, hub_cache):
    """Load the LM-head model + tokenizer ONCE for the whole group.

    The base ESMCModel has no LM head, so we load AutoModelForMaskedLM (the
    released biohub/ESMC-* checkpoints ship the head; EvolutionaryScale document
    it for zero-shot variant scoring). Returns (model, tokenizer, aa_token_ids).
    """
    from transformers import AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=hub_cache, local_files_only=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, cache_dir=hub_cache, local_files_only=True,
        attn_implementation=_attn_impl(),
    ).cuda().eval()
    return model, tokenizer, build_aa_token_ids(tokenizer)


def logits_batch(model, tokenizer, seqs, label):
    """Forward one length-bucketed micro-batch with the LM head. We request a
    special-tokens mask so real residue positions can be selected exactly per
    row — never assume a fixed BOS/EOS offset. Returns CPU float32 numpy arrays
    (logits, special_tokens_mask, attention_mask) of shape (rows, tokens, *).
    Logits cast back to fp32 on CPU so downstream LLR math is bf16-independent.
    """
    enc = tokenizer(
        seqs, return_tensors="pt", padding=True,
        return_special_tokens_mask=True,
    )
    inputs = {k: v.to(model.device)
              for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=AUTOCAST_DTYPE, enabled=AUTOCAST_DTYPE is not None
    ):
        outputs = model(**inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"BENCH_INFER_S {label} rows={len(seqs)} "
          f"{time.perf_counter() - _t0:.4f}", flush=True)
    return (
        outputs.logits.float().cpu().numpy(),
        enc["special_tokens_mask"].cpu().numpy(),
        enc["attention_mask"].cpu().numpy(),
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


def _gather_group(input_dirs, output_dirs):
    """Flatten a group of (chunk_dir, output_dir) into parallel lists keyed by a
    unique tag per sequence. The tag namespaces by chunk index so identical
    stems in different chunks never collide, and carries the chunk's own output
    dir so each result is scattered back to its origin chunk."""
    tags, seqs, out_by_tag, name_by_tag, len_by_tag = [], [], {}, {}, {}
    for ci, (in_dir, out_dir) in enumerate(zip(input_dirs, output_dirs)):
        for name, seq in parse_inputs(in_dir).items():
            tag = f"{ci}/{name}"
            tags.append(tag)
            seqs.append(seq)
            out_by_tag[tag] = out_dir
            name_by_tag[tag] = name
            len_by_tag[tag] = len(seq)
    return tags, seqs, out_by_tag, name_by_tag, len_by_tag


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    # Group persistence: one process loads the model ONCE and serves every
    # chunk in --input-dirs (parallel to --output-dirs). Single-chunk runs pass
    # one of each, reproducing the pre-grouping behaviour exactly.
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dirs", nargs="+", required=True)
    parser.add_argument("--size", type=str, default="6B", choices=[*ESMC_MODELS, "all"])
    parser.add_argument("--token-budget", type=int, default=None,
                        help="Padded-token budget per micro-batch (rows * max_len). "
                             "Default: per-size DEFAULT_TOKEN_BUDGET.")
    parser.add_argument(
        "--save-logits", action="store_true",
        help="Produce per-residue vocabulary logits (logits.npy + "
             "aa_token_ids.json) for zero-shot mutation scanning instead of "
             "embeddings. Loads the LM-head model (AutoModelForMaskedLM).",
    )
    args = parser.parse_args()
    if len(args.input_dirs) != len(args.output_dirs):
        parser.error("--input-dirs and --output-dirs must be equal length")
    hub_cache = _enforce_offline(args.cache)
    AUTOCAST_DTYPE = _enable_tf32()

    tags, sequences, out_by_tag, name_by_tag, len_by_tag = _gather_group(
        args.input_dirs, args.output_dirs)
    seq_by_tag = dict(zip(tags, sequences))

    sizes = list(ESMC_MODELS) if args.size == "all" else [args.size]

    for size in sizes:
        model_name = ESMC_MODELS[size]
        budget = args.token_budget or DEFAULT_TOKEN_BUDGET.get(size, 16000)
        if args.save_logits:
            model, tokenizer, aa_token_ids = load_logits_model(model_name, hub_cache)
            pad_free = getattr(model.config, "_attn_implementation", None) == "flash_attention_2"
            for batch_tags in token_budget_batches(tags, sequences, budget, pad_free):
                seqs = [seq_by_tag[t] for t in batch_tags]
                logits, special_mask, attn_mask = logits_batch(
                    model, tokenizer, seqs, f"{model_name}:{size}")
                for row, tag in enumerate(batch_tags):
                    rows = _real_residue_logits(
                        logits, special_mask, attn_mask, row,
                        len_by_tag[tag], name_by_tag[tag])
                    save_logits_outputs(out_by_tag[tag], name_by_tag[tag],
                                        size, rows, aa_token_ids)
        else:
            model, tokenizer = load_embed_model(model_name, hub_cache)
            pad_free = getattr(model.config, "_attn_implementation", None) == "flash_attention_2"
            for batch_tags in token_budget_batches(tags, sequences, budget, pad_free):
                seqs = [seq_by_tag[t] for t in batch_tags]
                hidden = embed_batch(model, tokenizer, seqs, f"{model_name}:{size}")
                for row, tag in enumerate(batch_tags):
                    h = hidden[row, :len_by_tag[tag]].cpu()
                    save_outputs(out_by_tag[tag], name_by_tag[tag], size, h)
