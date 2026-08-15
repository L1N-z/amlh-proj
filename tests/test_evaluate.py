import math

import pytest
from sklearn.metrics import f1_score

from amlh.evaluate import (
    accuracy_at_k,
    accuracy_coverage_curve,
    pairwise_top1_disagreement,
    score_ranked,
)

RANKED = [
    ["a", "b", "c"],
    ["b", "a", "c"],
    ["c", "b", "a"],
    ["a", "c", "b"],
]
GOLD = ["a", "a", "a", "b"]


def test_accuracy_at_k_basic():
    assert accuracy_at_k(RANKED, GOLD, 1) == pytest.approx(0.25)
    assert accuracy_at_k(RANKED, GOLD, 2) == pytest.approx(0.5)
    assert accuracy_at_k(RANKED, GOLD, 3) == pytest.approx(1.0)


def test_score_ranked_regression_matches_independent_oracle():
    """Pins score_ranked's output against a hand-computed oracle that does not
    call accuracy_at_k, so the accuracy_at_k extraction can't silently change
    score_ranked's behaviour."""
    top1 = [r[0] for r in RANKED]
    expected_accuracy = sum(p == g for p, g in zip(top1, GOLD)) / len(GOLD)
    expected_acc_at_5 = sum(g in r[:5] for r, g in zip(RANKED, GOLD)) / len(GOLD)
    expected_macro_f1 = f1_score(GOLD, top1, average="macro", zero_division=0)
    expected_mrr = sum(
        (1.0 / (r.index(g) + 1) if g in r else 0.0) for r, g in zip(RANKED, GOLD)
    ) / len(GOLD)

    expected = {
        "accuracy": expected_accuracy,
        "acc_at_5": expected_acc_at_5,
        "macro_f1": expected_macro_f1,
        "mrr": expected_mrr,
    }
    assert score_ranked(RANKED, GOLD) == pytest.approx(expected)


def test_pairwise_top1_disagreement():
    top1 = {"a": ["x", "y", "z"], "b": ["x", "z", "z"], "c": ["q", "y", "z"]}
    matrix = pairwise_top1_disagreement(top1)
    assert matrix.loc["a", "b"] == matrix.loc["b", "a"] == 1
    assert matrix.loc["a", "c"] == matrix.loc["c", "a"] == 1
    assert matrix.loc["b", "c"] == matrix.loc["c", "b"] == 2
    assert matrix.loc["a", "a"] == 0


def test_accuracy_coverage_curve():
    top_sim = [0.9, 0.8, 0.5, 0.95]
    df = accuracy_coverage_curve(RANKED, GOLD, top_sim, [0.0, 0.85, 0.99])

    assert list(df["coverage"]) == pytest.approx([1.0, 0.5, 0.0])
    assert list(df["n_covered"]) == [4, 2, 0]
    assert df["accuracy"].iloc[0] == pytest.approx(0.25)
    assert df["accuracy"].iloc[1] == pytest.approx(0.5)
    assert math.isnan(df["accuracy"].iloc[2])
