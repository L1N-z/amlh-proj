"""Frozen test-run logic for `05_results`.

Everything here runs *after* every hyperparameter in `config.py` is frozen. Nothing in
this module selects anything, and no function returns a quantity that could be fed back
into a choice — that is the whole point of confining the test set to this stage.

The test set is reached only through `data.load_test`, which drops `answer` at the
loader, so hard rule #1 is enforced before any code in this module sees a row.
`assert_no_answer_column` re-checks it anyway at each entry point, because the cost of
the check is nothing and the cost of being wrong is the coursework.

Two of the three arms need a GPU, so the test run is split across `05_results.ipynb`
(CPU: Arm 1 + all aggregation) and `05_results_colab.ipynb` (GPU: Arms 2 and 3). The
Colab side rebuilds the Arm 1 test shortlist itself and `assert_reproduces_arm1_test`
proves it matches the CPU side item for item before any prompt is sent — the same guard
that would have caught the 2026-08-23 QLA incident.
"""

from __future__ import annotations

import pandas as pd

from amlh import arm1_experiments as ae
from amlh import evaluate
from amlh.config import ARTEFACTS_DIR, HYPERPARAMETERS

SHORTLIST_DEPTH = 20
"""Ranking depth for the persisted test predictions.

Matches `02_arm1.ipynb`'s `SHORTLIST_DEPTH`, so the test frames carry the same
`top_1..top_20` columns as `arm1_val_predictions.csv` and Arm 3 has a shortlist to
select from. `knn_rank`'s `depth` only appends below the head, so rank 1 is identical to
the `depth=None` path the Arm 1 grid reported.
"""

ARM_LABELS = {
    "arm1_tfidf_knn": "Arm 1 — TF-IDF + k-NN retrieval",
    "arm2_bio_clinicalbert": "Arm 2 — Bio_ClinicalBERT fine-tune",
    "arm3_llm_rerank": "Arm 3 — shortlist + MediPhi selection",
}

TEST_PREDICTION_FILES = {
    "arm1_tfidf_knn": "arm1_test_predictions.csv",
    "arm2_bio_clinicalbert": "arm2_test_predictions.csv",
    "arm3_llm_rerank": "arm3_test_predictions.csv",
}


def assert_no_answer_column(df: pd.DataFrame, name: str = "test") -> None:
    """Hard rule #1, re-checked at every entry point.

    `data.load_test` already drops `answer`, so this should never fire. It exists
    because a future edit that loads the test CSV directly would otherwise fail
    silently and invisibly, and the report would be built on a leaked model.
    """
    if "answer" in df.columns:
        raise ValueError(
            f"{name} frame carries an `answer` column. Test-side answers are never an "
            "inference-time input — load via data.load_test, which drops it."
        )


def build_test_predictions(fit_df: pd.DataFrame, test_df: pd.DataFrame, hp=HYPERPARAMETERS) -> pd.DataFrame:
    """Arm 1's frozen top-1 and top-20 shortlist over the test questions.

    Delegates to `arm1_experiments.build_val_predictions`, which is query-frame agnostic
    despite its name — reusing it is what guarantees the test path and the validated
    path are the same code at the same hyperparameters, rather than two implementations
    that happen to agree today.

    `fit_df` is `split_fit`, not the full training set: CLAUDE.md fixes the tested model
    as the validated model so the validation→test optimism analysis describes one object.
    """
    assert_no_answer_column(test_df)
    return ae.build_val_predictions(fit_df, test_df, hp, depth=SHORTLIST_DEPTH)


def assert_reproduces_arm1_test(
    shortlists: list[list[str]], reference_path=None, tolerance: int = 0
) -> None:
    """Prove a Colab-rebuilt Arm 1 test shortlist matches the CPU-side one item for item.

    The Arm 3 incident of 2026-08-23 was invisible precisely because a degraded index
    still produced plausible shortlists. Rank 1 is a sharp fingerprint of the index that
    built it, so comparing it against the persisted CPU run catches a missing `D`
    component before any GPU time is spent.

    TF-IDF and NearestNeighbors are deterministic on identical input, so the default
    `tolerance=0` is correct; the parameter exists only to make a deliberate relaxation
    explicit rather than silent.
    """
    reference_path = reference_path or ARTEFACTS_DIR / TEST_PREDICTION_FILES["arm1_tfidf_knn"]
    reference = pd.read_csv(reference_path)
    expected = reference["pred"].tolist()
    if len(shortlists) != len(expected):
        raise ValueError(f"rebuilt {len(shortlists)} shortlists, reference holds {len(expected)} rows")

    mismatches = [i for i, (s, e) in enumerate(zip(shortlists, expected)) if s[0] != e]
    if len(mismatches) > tolerance:
        raise ValueError(
            f"rebuilt Arm 1 test shortlist disagrees with {reference_path.name} on "
            f"{len(mismatches)}/{len(expected)} items (tolerance {tolerance}). The index is "
            "not the frozen one — check doc coverage and the QLAD variant before proceeding."
        )
    print(f"shortlist reproduces frozen Arm 1 test top-1 on {len(expected) - len(mismatches)}/{len(expected)} items")


def load_available_arms(artefacts_dir=ARTEFACTS_DIR) -> dict[str, pd.DataFrame]:
    """Load whichever arms' test predictions have been produced so far.

    Returns only what exists. `05_results.ipynb` must run standalone after a kernel
    restart, and on a machine where the Colab half has not been run yet that means
    reporting Arm 1 alone and saying so — not raising.
    """
    frames = {}
    for arm, filename in TEST_PREDICTION_FILES.items():
        path = artefacts_dir / filename
        if path.exists():
            frames[arm] = pd.read_csv(path)
    return frames


def assert_query_aligned(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Refuse to compare arms whose rows are not the same items in the same order.

    McNemar is only meaningful on matched pairs, and these frames are written by two
    notebooks on two machines. Compares gold labels position by position rather than
    trusting row order implicitly.
    """
    reference_name, reference = next(iter(frames.items()))
    gold = reference["gold"].tolist()
    for name, frame in frames.items():
        if len(frame) != len(gold):
            raise ValueError(f"{name} has {len(frame)} rows, {reference_name} has {len(gold)}")
        if frame["gold"].tolist() != gold:
            raise ValueError(f"{name} gold labels are not aligned with {reference_name}")
    return gold


def score_arms(frames: dict[str, pd.DataFrame], seed: int | None = None) -> pd.DataFrame:
    """Headline test accuracy per arm, with bootstrap 95% CI and the ranked metrics.

    Accuracy is the headline metric the brief specifies. Top-5 and MRR ride along
    because the error analysis needs them to explain why accuracy is capped, not because
    they are being promoted to headline status.
    """
    from amlh.config import SEED

    seed = SEED if seed is None else seed
    gold = assert_query_aligned(frames)
    top_cols = [f"top_{i}" for i in range(1, SHORTLIST_DEPTH + 1)]

    rows = []
    for arm, frame in frames.items():
        pred = frame["pred"].tolist()
        ci = evaluate.bootstrap_accuracy_ci(pred, gold, seed=seed)
        row = {
            "arm": arm,
            "label": ARM_LABELS.get(arm, arm),
            "n": len(gold),
            "accuracy": sum(p == g for p, g in zip(pred, gold)) / len(gold),
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
        }
        available = [c for c in top_cols if c in frame.columns]
        if available:
            ranked = [[lab for lab in r if isinstance(lab, str)] for r in frame[available].values.tolist()]
            scored = evaluate.score_ranked(ranked, gold)
            row.update({"acc_at_5": scored["acc_at_5"], "macro_f1": scored["macro_f1"], "mrr": scored["mrr"]})
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_mcnemar(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every pairwise McNemar over the matched test items.

    Reported for all pairs because no final system is nominated — CLAUDE.md's test-scope
    decision is that the arms are compared side by side, so every comparison is a result
    rather than a step towards one.
    """
    from itertools import combinations

    gold = assert_query_aligned(frames)
    rows = []
    for a, b in combinations(frames, 2):
        result = evaluate.mcnemar_exact(frames[a]["pred"].tolist(), frames[b]["pred"].tolist(), gold)
        rows.append({"system_a": a, "system_b": b, **result})
    return pd.DataFrame(rows)


def validation_test_gap(
    test_scores: pd.DataFrame, val_comparison_path=None
) -> pd.DataFrame:
    """Pair each arm's validation accuracy against its test accuracy.

    Quantifies the optimism CLAUDE.md records at roughly 35pp for Arm 1, now measured
    per arm rather than assumed to be uniform across them. This is a diagnostic of the
    hold-out protocol, computed after the test run; it selects nothing.
    """
    val_comparison_path = val_comparison_path or ARTEFACTS_DIR / "cross_arm_val_comparison.csv"
    val = pd.read_csv(val_comparison_path)[["arm", "accuracy"]].rename(columns={"accuracy": "val_accuracy"})
    merged = test_scores[["arm", "label", "accuracy"]].rename(columns={"accuracy": "test_accuracy"})
    merged = merged.merge(val, on="arm", how="left")
    merged["optimism"] = merged["val_accuracy"] - merged["test_accuracy"]
    return merged


def confusion_pairs(frame: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Most frequent (gold, predicted) error pairs, with a same-family flag.

    The family flag matters because CLAUDE.md records 355/906 labels sharing a prefix
    family; an error inside `baby_*` is a different kind of failure from one across
    unrelated conditions, and collapsing them would hide the actual error mode.
    """
    errors = frame[frame["pred"] != frame["gold"]]
    counts = (
        errors.groupby(["gold", "pred"]).size().reset_index(name="n").sort_values("n", ascending=False)
    )
    counts["same_family"] = [
        g.split("_")[0] == p.split("_")[0] for g, p in zip(counts["gold"], counts["pred"])
    ]
    return counts.head(top_n).reset_index(drop=True)


def rerank_decomposition(arm3_frame: pd.DataFrame) -> pd.DataFrame:
    """Split Arm 3's items by what the LLM actually did to the shortlist it was handed.

    Arm 3's headline accuracy is a blend of three populations that behave completely
    differently, and reporting it without the split makes the arm look uniformly weak
    rather than weak in one specific, diagnosable way:

    - `kept_rank_1`  — the LLM agreed with the retriever; accuracy is Arm 1's by definition.
    - `fell_back`    — parsing found no shortlist label, so Arm 1's top-1 was returned
                       unchanged. Inert by construction, neither helps nor hurts.
    - `moved_off_rank_1` — the LLM overrode the retriever. On validation this is where
                       every net loss came from.

    `arm1_accuracy` on each stratum is the counterfactual: what the shortlist alone would
    have scored on those same items. The gap between the two columns on
    `moved_off_rank_1` is the intervention's real cost or benefit.
    """
    frame = arm3_frame.copy()
    correct = frame["pred"] == frame["gold"]
    arm1_correct = frame["arm1_pred"] == frame["gold"]

    fell_back = frame["fallback_fired"].astype(bool)
    moved = (~fell_back) & (frame["parsed_rank"] > 1)
    kept = (~fell_back) & (frame["parsed_rank"] == 1)

    rows = []
    for name, mask in (("kept_rank_1", kept), ("fell_back", fell_back), ("moved_off_rank_1", moved)):
        n = int(mask.sum())
        rows.append(
            {
                "stratum": name,
                "n": n,
                "share": n / len(frame) if len(frame) else 0.0,
                "arm3_accuracy": float(correct[mask].mean()) if n else 0.0,
                "arm1_accuracy": float(arm1_correct[mask].mean()) if n else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    out["accuracy_delta"] = out["arm3_accuracy"] - out["arm1_accuracy"]

    moved_n = int(moved.sum())
    gained = int((moved & correct & ~arm1_correct).sum())
    lost = int((moved & ~correct & arm1_correct).sum())
    print(
        f"interventions: {moved_n} | rescued {gained} | broke {lost} | "
        f"precision {gained / moved_n:.3f}" if moved_n else "no interventions"
    )
    return out


def worked_examples(frames: dict[str, pd.DataFrame], n: int = 5, seed: int | None = None) -> pd.DataFrame:
    """Sample correct and incorrect test predictions per arm for the report.

    The brief asks for "examples of correct and incorrect predictions per method". Drawn
    with a fixed seed so the report's examples are stable across reruns.
    """
    from amlh.config import SEED

    seed = SEED if seed is None else seed
    rows = []
    for arm, frame in frames.items():
        correct = frame[frame["pred"] == frame["gold"]]
        wrong = frame[frame["pred"] != frame["gold"]]
        for outcome, subset in (("correct", correct), ("incorrect", wrong)):
            take = subset.sample(min(n, len(subset)), random_state=seed)
            for row in take.itertuples(index=False):
                rows.append(
                    {
                        "arm": arm,
                        "outcome": outcome,
                        "question": row.question,
                        "gold": row.gold,
                        "pred": row.pred,
                    }
                )
    return pd.DataFrame(rows)
