import math

import pytest
from sklearn.metrics import f1_score

from amlh.evaluate import (
    accuracy_at_k,
    accuracy_coverage_curve,
    mcnemar_exact,
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


def test_mcnemar_exact_counts_only_discordant_pairs():
    gold = ["a", "a", "a", "a", "b", "b"]
    # items 0,1 both right; item 2 both wrong -> all concordant, no evidence.
    # items 3,4 only A right; item 5 only B right.
    pred_a = ["a", "a", "x", "a", "b", "x"]
    pred_b = ["a", "a", "x", "x", "x", "b"]
    result = mcnemar_exact(pred_a, pred_b, gold)

    assert result["only_a_correct"] == 2
    assert result["only_b_correct"] == 1
    assert result["n_discordant"] == 3
    assert result["accuracy_a"] == pytest.approx(4 / 6)
    assert result["accuracy_b"] == pytest.approx(3 / 6)
    assert result["accuracy_diff"] == pytest.approx(1 / 6)


def test_mcnemar_exact_p_value_matches_hand_computed_binomial():
    # 3 discordant pairs, split 2/1. Two-sided exact p = 2 * P(X <= 1 | n=3, p=0.5)
    # = 2 * (1 + 3)/8 = 1.0 -- capped at 1.
    gold = ["a", "a", "a"]
    pred_a = ["a", "a", "x"]
    pred_b = ["x", "x", "a"]
    assert mcnemar_exact(pred_a, pred_b, gold)["p_value"] == pytest.approx(1.0)

    # 6 discordant pairs, all favouring A. Two-sided exact p = 2 * (1/64) = 0.03125.
    gold = ["a"] * 6
    pred_a = ["a"] * 6
    pred_b = ["x"] * 6
    assert mcnemar_exact(pred_a, pred_b, gold)["p_value"] == pytest.approx(2 / 64)


def test_mcnemar_exact_identical_systems_are_unresolved():
    gold = ["a", "b", "c"]
    pred = ["a", "b", "x"]
    result = mcnemar_exact(pred, list(pred), gold)
    assert result["n_discordant"] == 0
    assert result["p_value"] == 1.0
    assert result["accuracy_diff"] == 0.0


def test_mcnemar_exact_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        mcnemar_exact(["a", "b"], ["a"], ["a", "b"])
    with pytest.raises(ValueError):
        mcnemar_exact([], [], [])


def test_accuracy_coverage_curve():
    top_sim = [0.9, 0.8, 0.5, 0.95]
    df = accuracy_coverage_curve(RANKED, GOLD, top_sim, [0.0, 0.85, 0.99])

    assert list(df["coverage"]) == pytest.approx([1.0, 0.5, 0.0])
    assert list(df["n_covered"]) == [4, 2, 0]
    assert df["accuracy"].iloc[0] == pytest.approx(0.25)
    assert df["accuracy"].iloc[1] == pytest.approx(0.5)
    assert math.isnan(df["accuracy"].iloc[2])
