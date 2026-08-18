"""Scoring for ranked disease predictions, plus paired system comparison.

`score_ranked` and friends score a single ranked run; `mcnemar_exact` compares
two runs over the same query set; `bootstrap_accuracy_ci` gives a single run's
accuracy a confidence interval.
"""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from amlh.config import SEED


def accuracy_at_k(ranked: list[list[str]], gold: list[str], k: int) -> float:
    """Fraction of queries where gold appears in the top-k predicted labels."""
    return sum(g in r[:k] for r, g in zip(ranked, gold)) / len(gold)


def score_ranked(ranked: list[list[str]], gold: list[str]) -> dict[str, float]:
    """ranked[i] = query i's predicted labels, best first. gold[i] = true label."""
    top1 = [r[0] if r else None for r in ranked]

    accuracy = accuracy_at_k(ranked, gold, 1)
    acc_at_5 = accuracy_at_k(ranked, gold, 5)
    macro_f1 = f1_score(gold, top1, average="macro", zero_division=0)

    def _rr(r, g):
        return 1.0 / (r.index(g) + 1) if g in r else 0.0

    mrr = sum(_rr(r, g) for r, g in zip(ranked, gold)) / len(gold)

    return {"accuracy": accuracy, "acc_at_5": acc_at_5, "macro_f1": macro_f1, "mrr": mrr}


def mcnemar_exact(
    pred_a: list[str], pred_b: list[str], gold: list[str]
) -> dict[str, float]:
    """Exact McNemar test on two systems' top-1 predictions over the same,
    query-aligned items.

    The comparison is paired — both systems predict the same queries — so items
    the two agree on carry no evidence about which is better, and the test
    conditions on the discordant pairs alone. That is why a difference of two
    accuracies must not be judged against a single-proportion standard error:
    that SE ignores the pairing and answers a different question.

    The exact binomial tail is used rather than the chi-square approximation
    because the discordant count over a 200-item hold-out is small, and the
    approximation is unreliable below ~25 discordant pairs. Under the null the
    binomial is symmetric (p=0.5), so the two-sided p-value is twice the lower
    tail, capped at 1.

    Returns the discordant counts alongside the p-value so the report can quote
    the pair counts the test actually consumed, not just its verdict.
    """
    if not len(pred_a) == len(pred_b) == len(gold):
        raise ValueError("pred_a, pred_b and gold must be query-aligned and the same length")
    if not gold:
        raise ValueError("cannot compare systems over an empty query set")

    correct_a = [p == g for p, g in zip(pred_a, gold)]
    correct_b = [p == g for p, g in zip(pred_b, gold)]
    only_a_correct = sum(a and not b for a, b in zip(correct_a, correct_b))
    only_b_correct = sum(b and not a for a, b in zip(correct_a, correct_b))
    n_discordant = only_a_correct + only_b_correct

    if n_discordant == 0:
        p_value = 1.0
    else:
        lower_tail = sum(
            math.comb(n_discordant, i) for i in range(min(only_a_correct, only_b_correct) + 1)
        )
        p_value = min(1.0, 2 * lower_tail / 2**n_discordant)

    n = len(gold)
    return {
        "n": n,
        "accuracy_a": sum(correct_a) / n,
        "accuracy_b": sum(correct_b) / n,
        "accuracy_diff": (sum(correct_a) - sum(correct_b)) / n,
        "only_a_correct": only_a_correct,
        "only_b_correct": only_b_correct,
        "n_discordant": n_discordant,
        "p_value": p_value,
    }


def bootstrap_accuracy_ci(
    pred: list[str],
    gold: list[str],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, float]:
    """Percentile bootstrap 95% CI for one system's top-1 accuracy.

    Resamples *items*, not classes or predictions in isolation, because the
    estimand is overall accuracy over the hold-out and each item is an iid
    draw from it — resampling any other unit would answer a different
    question. Uses a local `numpy.random.default_rng(seed)` rather than
    global NumPy state, so this stochastic stage cannot perturb the RNG
    surrounding code depends on; re-seed explicitly if calling it more than
    once in a pipeline.

    When every item is correct (or every item is wrong), every bootstrap
    resample gives the same accuracy, so `ci_low == ci_high == accuracy`.
    That is the correct behaviour, not a bug: the bootstrap has zero
    resampling variance to express as uncertainty. Arm 1's per-class slices
    are small enough to hit this routinely.
    """
    if len(pred) != len(gold):
        raise ValueError("pred and gold must be query-aligned and the same length")
    if not gold:
        raise ValueError("cannot compute a confidence interval over an empty query set")

    n = len(gold)
    correct = np.array([p == g for p, g in zip(pred, gold)], dtype=float)

    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, n, size=(n_boot, n))
    boot_accuracies = correct[resample_idx].mean(axis=1)
    ci_low, ci_high = np.quantile(boot_accuracies, [alpha / 2, 1 - alpha / 2])

    return {
        "accuracy": correct.mean(),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
        "n_boot": n_boot,
    }


def pairwise_top1_disagreement(top1_by_key: dict[str, list[str]]) -> pd.DataFrame:
    """Symmetric matrix (keys x keys) of how many items differ in their top-1
    prediction between each pair of keys (e.g. index variants run over the
    same query set). Diagonal is 0. All lists must be the same length and in
    query-aligned order."""
    keys = list(top1_by_key)
    matrix = pd.DataFrame(0, index=keys, columns=keys)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            changed = sum(x != y for x, y in zip(top1_by_key[a], top1_by_key[b]))
            matrix.loc[a, b] = changed
            matrix.loc[b, a] = changed
    return matrix


def accuracy_coverage_curve(
    ranked: list[list[str]],
    gold: list[str],
    top_sim: list[float],
    thresholds: list[float],
) -> pd.DataFrame:
    """Top-1 accuracy restricted to queries whose top similarity clears each
    threshold. `accuracy` is NaN where the threshold covers no queries."""
    rows = []
    n = len(gold)
    for t in thresholds:
        kept = [i for i in range(n) if top_sim[i] >= t]
        n_covered = len(kept)
        coverage = n_covered / n
        if n_covered == 0:
            accuracy = math.nan
        else:
            correct = sum(ranked[i][0] == gold[i] if ranked[i] else False for i in kept)
            accuracy = correct / n_covered
        rows.append(
            {"threshold": t, "coverage": coverage, "n_covered": n_covered, "accuracy": accuracy}
        )
    return pd.DataFrame(rows)
