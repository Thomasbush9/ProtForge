#!/usr/bin/env python3
"""
Zero-shot saturation-mutagenesis scoring from ESM-C logits.

Given the per-residue vocabulary logits of a wild-type sequence (one ESM-C
forward pass, saved by run_batch_esmc.py with --save-logits), compute a
wt-marginal log-likelihood ratio (LLR) for every single substitution:

    LLR(i, a) = log p(x_i = a | seq) - log p(x_i = wt_i | seq)

Following Meier et al. 2021 ("Language models enable zero-shot prediction of
the effects of mutations on protein function"). Negative LLR = the model finds
the mutation less likely than wild-type (often destabilizing/deleterious);
positive = more likely than wild-type.

Pure NumPy — no torch, no GPU. The model-side work (a logits.npy per sequence)
happens once in the ESM-C container; this scoring is a cheap CPU derivation, so
it can run in a post-step or in the webapp.

Key simplification: LLR is a *difference* of two log-softmax values at the same
position, so the softmax normalizer log(Z_i) cancels. We therefore work
directly on raw logits — no softmax needed for the ratio:

    LLR(i, a) = logit(i, a) - logit(i, wt_i)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# The 20 canonical amino acids, in a fixed column order for the output matrix.
CANONICAL_AAS = "ACDEFGHIKLMNPQRSTVWY"


def wt_marginal_llr(
    logits: np.ndarray,
    wt_seq: str,
    aa_token_ids: dict[str, int],
    aa_order: str = CANONICAL_AAS,
) -> np.ndarray:
    """Wild-type-marginal LLR matrix for a sequence.

    Args:
        logits: (L, V) raw logits for the L real residue positions only
            (special tokens such as BOS/EOS already removed), V = vocab size.
        wt_seq: the wild-type sequence, length L (each residue must be a key in
            aa_token_ids).
        aa_token_ids: maps each amino-acid letter to its column index in
            `logits` (the tokenizer's vocab id). Must cover every residue in
            wt_seq and every AA in aa_order.
        aa_order: amino acids forming the output columns (default: 20 canonical).

    Returns:
        (L, 20) array; entry [i, j] is the LLR of substituting position i with
        aa_order[j]. The wild-type's own column at each position is exactly 0.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (L, V); got shape {logits.shape}")
    L, V = logits.shape
    if len(wt_seq) != L:
        raise ValueError(
            f"wt_seq length {len(wt_seq)} != logits rows {L} — check that special "
            "tokens were stripped and the sequence matches."
        )

    missing = (set(wt_seq) | set(aa_order)) - set(aa_token_ids)
    if missing:
        raise KeyError(f"aa_token_ids is missing token ids for: {sorted(missing)}")

    mut_cols = np.array([aa_token_ids[a] for a in aa_order])      # (20,)
    wt_cols = np.array([aa_token_ids[a] for a in wt_seq])          # (L,)
    if mut_cols.max() >= V or wt_cols.max() >= V:
        raise ValueError("aa_token_ids references a column outside the logits vocab.")

    mut_logits = logits[:, mut_cols]                               # (L, 20)
    wt_logits = logits[np.arange(L), wt_cols][:, None]             # (L, 1)
    return mut_logits - wt_logits


def position_sensitivity(llr: np.ndarray, wt_seq: str | None = None,
                         aa_order: str = CANONICAL_AAS) -> np.ndarray:
    """Per-position mean LLR over the non-wild-type substitutions.

    Lower (more negative) = the position is less tolerant to mutation. The
    wild-type column is 0, so summing the row and dividing by the number of
    non-wild-type columns gives their mean.

    That count is len(aa_order) - 1 only when the wild type is one of the
    columns. A residue outside aa_order — selenocysteine in a selenoprotein —
    has no zero column, so all len(aa_order) entries are real substitutions and
    dividing by one fewer would inflate the row. Pass `wt_seq` to get the
    per-row count right; without it the uniform len(aa_order) - 1 is assumed.
    """
    n = llr.shape[1]
    if wt_seq is None:
        return llr.sum(axis=1) / max(n - 1, 1)

    non_wt = np.array(
        [n - 1 if wt in aa_order else n for wt in wt_seq], dtype=np.float64
    )
    return llr.sum(axis=1) / np.maximum(non_wt, 1.0)


def write_scan_csv(path: Path, wt_seq: str, llr: np.ndarray,
                   aa_order: str = CANONICAL_AAS) -> Path:
    """Write the LLR matrix as a tidy-wide CSV: one row per position.

    Columns: position (1-based), wt_aa, sensitivity (mean non-wt LLR), then one
    column per amino acid in aa_order.
    """
    path = Path(path)
    sens = position_sensitivity(llr, wt_seq, aa_order)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "wt_aa", "sensitivity", *aa_order])
        for i, wt in enumerate(wt_seq):
            row = [i + 1, wt, f"{sens[i]:.4f}", *(f"{v:.4f}" for v in llr[i])]
            w.writerow(row)
    return path


def summarize_scan(wt_seq: str, llr: np.ndarray, top_n: int = 10,
                   aa_order: str = CANONICAL_AAS) -> dict:
    """Headline numbers for a scan: most/least mutation-tolerant positions and
    the best-scoring single substitutions. Used for the ranking table + summary.
    """
    sens = position_sensitivity(llr, wt_seq, aa_order)
    order = np.argsort(sens)  # ascending: most sensitive first
    least_tolerant = [
        {"position": int(i + 1), "wt_aa": wt_seq[i], "sensitivity": float(sens[i])}
        for i in order[:top_n]
    ]
    most_tolerant = [
        {"position": int(i + 1), "wt_aa": wt_seq[i], "sensitivity": float(sens[i])}
        for i in order[::-1][:top_n]
    ]
    # Best single substitutions (highest LLR), excluding wild-type (LLR == 0).
    flat = llr.copy()
    L = flat.shape[0]
    wt_cols = [aa_order.index(wt_seq[i]) if wt_seq[i] in aa_order else -1 for i in range(L)]
    for i, c in enumerate(wt_cols):
        if c >= 0:
            flat[i, c] = -np.inf
    idx = np.argsort(flat, axis=None)[::-1][:top_n]
    best = []
    for k in idx:
        i, j = divmod(int(k), llr.shape[1])
        best.append({
            "position": i + 1, "wt_aa": wt_seq[i],
            "mut_aa": aa_order[j], "llr": float(llr[i, j]),
        })
    return {
        "length": int(L),
        "mean_sensitivity": float(sens.mean()),
        "least_tolerant_positions": least_tolerant,
        "most_tolerant_positions": most_tolerant,
        "top_substitutions": best,
    }
