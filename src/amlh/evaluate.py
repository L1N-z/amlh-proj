"""Scoring for ranked disease predictions.

No bootstrap CI or McNemar's test here yet — those apply to system
comparisons, and this module only scores a single ranked run.
"""

import math

import pandas as pd
from sklearn.metrics import f1_score


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
