"""
Tests for the zero-shot saturation-mutagenesis scorer
(workflow/scripts/mutation_scan.py).

Pure NumPy, no model needed — synthetic logits with a known structure verify
the LLR math, the softmax-cancellation property, sensitivity, and the CSV.

Run with:
    PYTHONPATH=workflow/scripts python -m pytest webapp/tests/test_mutation_scan.py -v
"""

import csv

import numpy as np
import pytest

from mutation_scan import (
    wt_marginal_llr,
    position_sensitivity,
    write_scan_csv,
    summarize_scan,
)

# A tiny 4-token vocab: A,C,D,E -> columns 0..3 (no special tokens for the test).
AA = {"A": 0, "C": 1, "D": 2, "E": 3}
ORDER = "ACDE"


def test_llr_is_logit_difference_and_wt_is_zero():
    # 2 positions, wt = "AC". Logits chosen so differences are exact integers.
    logits = np.array([
        [2.0, 5.0, 1.0, 0.0],   # pos 0, wt=A (col 0, logit 2.0)
        [3.0, 4.0, 0.0, 1.0],   # pos 1, wt=C (col 1, logit 4.0)
    ])
    llr = wt_marginal_llr(logits, "AC", AA, aa_order=ORDER)
    assert llr.shape == (2, 4)
    # Wild-type column at each position must be exactly 0.
    assert llr[0, ORDER.index("A")] == 0.0
    assert llr[1, ORDER.index("C")] == 0.0
    # LLR = logit(mut) - logit(wt).
    assert llr[0, ORDER.index("C")] == pytest.approx(5.0 - 2.0)   # +3
    assert llr[0, ORDER.index("D")] == pytest.approx(1.0 - 2.0)   # -1
    assert llr[1, ORDER.index("A")] == pytest.approx(3.0 - 4.0)   # -1


def test_softmax_offset_cancels():
    # Adding a per-position constant to all logits must not change the LLR
    # (the softmax normalizer cancels in the difference).
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 4))
    wt = "ACDEA"
    base = wt_marginal_llr(logits, wt, AA, aa_order=ORDER)
    shifted = logits + rng.normal(size=(5, 1))  # per-row offset
    assert np.allclose(base, wt_marginal_llr(shifted, wt, AA, aa_order=ORDER))


def test_sensitivity_excludes_wt_zero():
    logits = np.array([[2.0, 5.0, 1.0, 0.0]])  # wt=A; LLRs: A=0,C=+3,D=-1,E=-2
    llr = wt_marginal_llr(logits, "A", AA, aa_order=ORDER)
    # mean over the 3 non-wt substitutions = (3 + -1 + -2)/3 = 0.0
    assert position_sensitivity(llr)[0] == pytest.approx(0.0)


def test_length_and_missing_token_guards():
    logits = np.zeros((2, 4))
    with pytest.raises(ValueError):
        wt_marginal_llr(logits, "ACD", AA, aa_order=ORDER)   # wrong length
    with pytest.raises(KeyError):
        wt_marginal_llr(logits, "AX", AA, aa_order=ORDER)    # X not in vocab map


def test_write_csv_and_summarize(tmp_path):
    logits = np.array([
        [2.0, 5.0, 1.0, 0.0],   # wt=A, best sub = C (+3)
        [3.0, 4.0, 0.0, 1.0],   # wt=C
    ])
    llr = wt_marginal_llr(logits, "AC", AA, aa_order=ORDER)

    csv_path = write_scan_csv(tmp_path / "scan.csv", "AC", llr, aa_order=ORDER)
    rows = list(csv.DictReader(open(csv_path)))
    assert rows[0]["position"] == "1" and rows[0]["wt_aa"] == "A"
    assert {"A", "C", "D", "E", "sensitivity"} <= set(rows[0])

    summ = summarize_scan("AC", llr, top_n=2, aa_order=ORDER)
    assert summ["length"] == 2
    # Best single substitution overall is A->C at position 1 (LLR +3).
    top = summ["top_substitutions"][0]
    assert (top["position"], top["mut_aa"]) == (1, "C")
    assert top["llr"] == pytest.approx(3.0)
