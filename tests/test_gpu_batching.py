"""Unit tests for the GPU-efficiency batching helpers in the container runners.

These cover the pure-logic pieces that decide correctness independently of any
GPU or model weights:

- ``token_budget_batches`` (run_batch_esmc.py / esmc_extract_sae.py): length
  bucketing must never drop or duplicate a sequence and must keep each batch's
  padded cost (rows * max_len) within budget unless a single sequence exceeds it.
- ``split_sae_outputs`` (esmc_extract_sae.py): re-splitting a batched, flattened
  sparse SAE output back to per-sequence per-residue activations. This is the
  high-risk index math (BOS at row 0, EOS at the last non-pad row, real residues
  in between, sequences concatenated in batch order) flagged during review.

The runner modules ``import torch`` at top, so we load them with a minimal torch
stub when torch is unavailable; the split test needs real tensor ops and is
skipped without torch.
"""

from pathlib import Path

import pytest

CONTAINERS = Path(__file__).resolve().parent.parent / "containers"


def _extract_funcs(filename: str, names):
    """Pull named top-level functions out of a runner file and exec them in an
    isolated namespace with `torch` available. The helpers under test are pure
    (stdlib + torch only), so this avoids importing the runner's heavy deps
    (transformers/numpy) on a CPU-only test host."""
    import ast
    src = (CONTAINERS / filename).read_text()
    tree = ast.parse(src)
    wanted = {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    missing = set(names) - set(wanted)
    assert not missing, f"{filename}: functions not found: {missing}"
    module = ast.Module(body=[wanted[n] for n in names], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {}
    try:
        import torch
        ns["torch"] = torch
    except ModuleNotFoundError:
        pass
    exec(compile(module, filename, "exec"), ns)
    return ns


def _batcher():
    ns = _extract_funcs("run_batch_esmc.py", ["token_budget_batches"])
    return ns["token_budget_batches"]


# --------------------------------------------------------------------------- #
# token_budget_batches
# --------------------------------------------------------------------------- #

def test_batcher_preserves_every_sequence():
    f = _batcher()
    names = [f"s{i}" for i in range(10)]
    seqs = ["A" * L for L in (150, 800, 140, 160, 900, 155, 170, 130, 145, 165)]
    batches = list(f(names, seqs, budget=2000))
    flat = [n for b in batches for n in b]
    assert sorted(flat) == sorted(names), "no sequence may be dropped or duplicated"


def test_batcher_respects_budget_except_single_outlier():
    f = _batcher()
    names = [f"s{i}" for i in range(6)]
    lens = [150, 160, 140, 155, 170, 800]
    seqs = ["A" * L for L in lens]
    budget = 2000
    for b in batches_for(f, names, seqs, budget):
        ls = [len(seqs[names.index(n)]) for n in b]
        padded_cost = len(b) * max(ls)
        # A batch may exceed budget ONLY when it is a single oversized sequence.
        assert padded_cost <= budget or len(b) == 1


def test_batcher_isolates_long_outlier():
    f = _batcher()
    # The 800-aa sequence (2*800=1600<2000 but pairing it with others overflows)
    # should not drag a big pad onto the short peptides.
    names = ["short1", "short2", "short3", "long"]
    seqs = ["A" * 150, "A" * 150, "A" * 150, "A" * 800]
    batches = list(f(names, seqs, budget=900))
    # the long one cannot share a batch under budget 900 (2*800 > 900)
    long_batch = next(b for b in batches if "long" in b)
    assert long_batch == ["long"], "long outlier must batch alone under a tight budget"


def test_batcher_length_sorted_within_run():
    f = _batcher()
    names = ["a", "b", "c"]
    seqs = ["A" * 300, "A" * 100, "A" * 200]
    # Budget large enough for all three in one batch; order should be by length.
    batches = list(f(names, seqs, budget=10_000))
    assert batches == [["b", "c", "a"]]


def batches_for(f, names, seqs, budget):
    return list(f(names, seqs, budget))


# --------------------------------------------------------------------------- #
# split_sae_outputs  (needs real torch)
# --------------------------------------------------------------------------- #

def test_split_sae_outputs_reconstructs_per_sequence():
    torch = pytest.importorskip("torch")
    split = _extract_funcs("esmc_extract_sae.py", ["split_sae_outputs"])["split_sae_outputs"]

    # Two sequences of lengths 3 and 2. Each contributes BOS + residues + EOS:
    # nonpad = [5, 4]. Flattened rows (batch-major): seq0 rows 0..4, seq1 rows 5..8.
    # Mark each row with a recognizable feature value = 100*seq + token_index so
    # we can assert the re-split picks exactly the inner residue rows.
    seq_lens = [3, 2]
    nonpad = [5, 4]
    n_feat = 4
    rows = []
    for si, npd in enumerate(nonpad):
        for t in range(npd):
            rows.append([100 * si + t] * n_feat)
    dense = torch.tensor(rows, dtype=torch.float32)
    sae_outputs = {"layer0": dense.to_sparse()}

    per_seq = split(sae_outputs, nonpad, seq_lens)
    assert len(per_seq) == 2

    # seq0: real residues are token indices 1,2,3 (drop BOS=0, EOS=4)
    s0 = per_seq[0]["layer0"]
    assert s0.shape == (3, n_feat)
    assert [int(s0[i, 0]) for i in range(3)] == [1, 2, 3]

    # seq1: real residues are token indices 1,2 (drop BOS=0, EOS=3), offset by seq0
    s1 = per_seq[1]["layer0"]
    assert s1.shape == (2, n_feat)
    assert [int(s1[i, 0]) for i in range(2)] == [101, 102]


def test_split_sae_outputs_rejects_row_count_mismatch():
    torch = pytest.importorskip("torch")
    split = _extract_funcs("esmc_extract_sae.py", ["split_sae_outputs"])["split_sae_outputs"]
    # 5 rows but nonpad claims 6 -> must fail loudly, never silently misalign.
    dense = torch.zeros((5, 3))
    with pytest.raises(SystemExit):
        split({"layer0": dense.to_sparse()}, nonpad_counts=[6], seq_lens=[4])


def test_split_sae_outputs_rejects_bad_bos_eos_count():
    torch = pytest.importorskip("torch")
    split = _extract_funcs("esmc_extract_sae.py", ["split_sae_outputs"])["split_sae_outputs"]
    # nonpad=5 but seq_len=2 implies seq_len+2=4 != 5 -> reject (token layout off).
    dense = torch.zeros((5, 3))
    with pytest.raises(SystemExit):
        split({"layer0": dense.to_sparse()}, nonpad_counts=[5], seq_lens=[2])
