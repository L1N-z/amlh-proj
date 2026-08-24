"""Cross-arm validation comparison: accuracy with bootstrap CI, plus every pairwise McNemar.

Validation only. Reads no test-set quantity. Exists because the three arms were each
selected in their own notebook, so nothing had yet compared them against one another on
matched items — and §3.2 of the report needs exactly that.

Arm 1 and Arm 2 both sit at 0.850 on the hold-out; a difference of two accuracies says
nothing about which is better when both systems answer the same 200 questions, so the
comparison has to be paired.

Writes `artefacts/cross_arm_val_comparison.csv` and `artefacts/cross_arm_val_mcnemar.csv`.
"""

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlh import evaluate  # noqa: E402
from amlh.config import ARTEFACTS_DIR, HYPERPARAMETERS, SEED  # noqa: E402


def load_arms() -> dict[str, pd.DataFrame]:
    """Load each arm's per-item validation predictions, query-aligned.

    Arm 3 is the frozen cell only — the selected prompt condition on the selected
    generator. The other five grid cells are reported for transparency elsewhere and
    are not systems.
    """
    arm1 = pd.read_csv(ARTEFACTS_DIR / "arm1_val_predictions.csv")
    arm2 = pd.read_csv(ARTEFACTS_DIR / "arm2_val_predictions.csv")

    arm3_all = pd.read_csv(ARTEFACTS_DIR / "arm3_val_predictions.csv")
    arm3 = arm3_all[
        (arm3_all["condition"] == HYPERPARAMETERS.prompt_mode)
        & (arm3_all["model_name"] == HYPERPARAMETERS.arm3_model_name)
    ].reset_index(drop=True)

    return {"arm1_tfidf_knn": arm1, "arm2_bio_clinicalbert": arm2, "arm3_llm_rerank": arm3}


def assert_query_aligned(arms: dict[str, pd.DataFrame]) -> list[str]:
    """McNemar is only valid on matched items, so refuse to run if the rows disagree.

    Checks gold labels position by position rather than trusting row order implicitly —
    the three arms were written by three different notebooks on two different machines.
    """
    reference_name, reference = next(iter(arms.items()))
    gold = reference["gold"].tolist()
    for name, frame in arms.items():
        if len(frame) != len(gold):
            raise ValueError(f"{name} has {len(frame)} rows, {reference_name} has {len(gold)}")
        if frame["gold"].tolist() != gold:
            raise ValueError(f"{name} gold labels are not aligned with {reference_name}")
    return gold


def main() -> None:
    arms = load_arms()
    gold = assert_query_aligned(arms)
    print(f"query-aligned on {len(gold)} validation items across {len(arms)} arms\n")

    rows = []
    for name, frame in arms.items():
        pred = frame["pred"].tolist()
        ci = evaluate.bootstrap_accuracy_ci(pred, gold, seed=SEED)
        accuracy = sum(p == g for p, g in zip(pred, gold)) / len(gold)
        rows.append(
            {
                "arm": name,
                "n": len(gold),
                "accuracy": accuracy,
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
            }
        )
    comparison = pd.DataFrame(rows)
    print(comparison.to_string(index=False))
    print()

    mcnemar_rows = []
    for a, b in combinations(arms, 2):
        result = evaluate.mcnemar_exact(arms[a]["pred"].tolist(), arms[b]["pred"].tolist(), gold)
        mcnemar_rows.append({"system_a": a, "system_b": b, **result})
    mcnemar = pd.DataFrame(mcnemar_rows)
    print(mcnemar.to_string(index=False))
    print()

    for row in mcnemar.itertuples(index=False):
        verdict = "RESOLVED" if row.p_value < 0.05 else "UNRESOLVED"
        print(f"{row.system_a} vs {row.system_b}: p = {row.p_value:.4g} -> {verdict}")

    comparison.to_csv(ARTEFACTS_DIR / "cross_arm_val_comparison.csv", index=False)
    mcnemar.to_csv(ARTEFACTS_DIR / "cross_arm_val_mcnemar.csv", index=False)
    print(f"\nwrote cross_arm_val_comparison.csv and cross_arm_val_mcnemar.csv to {ARTEFACTS_DIR}")


if __name__ == "__main__":
    main()
