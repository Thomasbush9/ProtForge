"""Tests for build_aa_token_ids in containers/run_batch_esmc.py.

The map decides which residues a mutation scan can score. Selenocysteine is the
case that matters: DIO3 and the other deiodinases carry a catalytic Sec, and
wt_marginal_llr needs its logit column for the LLR denominator — without it the
whole scan raises KeyError rather than scoring a single position.

The function is pure (stdlib only), so it is pulled out of the runner by AST
rather than importing the module, which would drag in torch/transformers.
"""

import ast
from pathlib import Path

import pytest

CONTAINERS = Path(__file__).resolve().parent.parent / "containers"
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"


def _build_aa_token_ids():
    """Extract the function plus the module constants it closes over."""
    tree = ast.parse((CONTAINERS / "run_batch_esmc.py").read_text())
    wanted_consts = {"CANONICAL_AAS", "WT_ONLY_AAS"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in wanted_consts for t in node.targets
        ):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "build_aa_token_ids":
            body.append(node)

    names = {getattr(n, "name", None) for n in body}
    assert "build_aa_token_ids" in names, "build_aa_token_ids not found in runner"

    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {}
    exec(compile(module, "run_batch_esmc.py", "exec"), ns)
    return ns["build_aa_token_ids"], ns["CANONICAL_AAS"], ns["WT_ONLY_AAS"]


class FakeTokenizer:
    """Minimal stand-in: a dict vocab, unknown tokens collapse to unk_token_id."""

    def __init__(self, vocab, unk_token_id=99):
        self.vocab = vocab
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]


def _vocab(letters):
    return {aa: i for i, aa in enumerate(letters)}


def test_selenocysteine_is_included_when_the_vocabulary_has_it():
    """The DIO3 case: without U, scoring a selenoprotein fails outright."""
    f, canonical, _ = _build_aa_token_ids()
    tok = FakeTokenizer(_vocab(canonical + "U"))
    ids = f(tok)
    assert "U" in ids
    assert ids["U"] == len(canonical)   # appended after the canonical 20


def test_all_twenty_canonical_residues_are_mapped():
    f, canonical, _ = _build_aa_token_ids()
    ids = f(FakeTokenizer(_vocab(canonical + "U")))
    assert set(canonical) <= set(ids)
    assert len(canonical) == 20


def test_optional_residue_is_dropped_rather_than_mapped_to_unk():
    """A vocabulary without Sec must still give a usable map for normal proteins,
    not one where U silently points at the unknown-token logit."""
    f, canonical, _ = _build_aa_token_ids()
    ids = f(FakeTokenizer(_vocab(canonical), unk_token_id=77))
    assert "U" not in ids
    assert set(ids) == set(canonical)


def test_missing_canonical_residue_is_fatal():
    """Silently scoring against the unk column would corrupt every LLR."""
    f, canonical, _ = _build_aa_token_ids()
    tok = FakeTokenizer(_vocab(canonical.replace("W", "")), unk_token_id=77)
    with pytest.raises(RuntimeError, match="canonical residue 'W'"):
        f(tok)


def test_sec_is_a_wild_type_residue_only_not_a_scan_target():
    """U must not leak into the substitution columns — Sec insertion needs a
    SECIS element, so 'mutate X to Sec' is not a meaningful prediction."""
    _, canonical, wt_only = _build_aa_token_ids()
    assert "U" in wt_only
    assert "U" not in canonical
