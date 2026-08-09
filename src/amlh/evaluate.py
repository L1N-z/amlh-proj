"""Scoring for ranked disease predictions.

No bootstrap CI or McNemar's test here yet — those apply to system
comparisons, and this module only scores a single ranked run.
"""

from sklearn.metrics import f1_score


def score_ranked(ranked: list[list[str]], gold: list[str]) -> dict[str, float]:
    """ranked[i] = query i's predicted labels, best first. gold[i] = true label."""
    n = len(gold)
    top1 = [r[0] if r else None for r in ranked]

    accuracy = sum(p == g for p, g in zip(top1, gold)) / n
    acc_at_5 = sum(g in r[:5] for r, g in zip(ranked, gold)) / n
    macro_f1 = f1_score(gold, top1, average="macro", zero_division=0)

    def _rr(r, g):
        return 1.0 / (r.index(g) + 1) if g in r else 0.0

    mrr = sum(_rr(r, g) for r, g in zip(ranked, gold)) / n

    return {"accuracy": accuracy, "acc_at_5": acc_at_5, "macro_f1": macro_f1, "mrr": mrr}
