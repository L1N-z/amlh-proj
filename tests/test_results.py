import pandas as pd
import pytest
from sklearn.metrics import f1_score

from amlh import results


def make_frame(preds, golds, questions=None):
    n = len(golds)
    frame = pd.DataFrame(
        {
            "question": questions or [f"q{i}" for i in range(n)],
            "gold": golds,
            "pred": preds,
        }
    )
    for i in range(results.SHORTLIST_DEPTH):
        frame[f"top_{i + 1}"] = preds if i == 0 else [None] * n
    return frame


GOLD = ["a_one", "a_two", "b_one", "b_two"]


def test_assert_no_answer_column_rejects_leaked_frame():
    """Hard rule #1's last line of defence. `data.load_test` already drops the column,
    so this only fires if a future edit reads the test CSV directly."""
    with pytest.raises(ValueError, match="answer"):
        results.assert_no_answer_column(pd.DataFrame({"question": ["q"], "answer": ["a"]}))


def test_assert_no_answer_column_accepts_clean_frame():
    results.assert_no_answer_column(pd.DataFrame({"question": ["q"], "disease": ["d"]}))


def test_assert_query_aligned_rejects_reordered_rows():
    """McNemar is only valid on matched pairs. Two frames can hold the same items in a
    different order and still have identical accuracy, so length checks alone would pass
    while every pairing was wrong."""
    a = make_frame(GOLD, GOLD)
    b = make_frame(GOLD[::-1], GOLD[::-1])
    with pytest.raises(ValueError, match="not aligned"):
        results.assert_query_aligned({"a": a, "b": b})


def test_assert_query_aligned_rejects_length_mismatch():
    with pytest.raises(ValueError, match="rows"):
        results.assert_query_aligned({"a": make_frame(GOLD, GOLD), "b": make_frame(GOLD[:2], GOLD[:2])})


def test_score_arms_accuracy_matches_hand_count():
    frames = {"arm_x": make_frame(["a_one", "a_two", "b_one", "WRONG"], GOLD)}
    scored = results.score_arms(frames)
    assert scored.loc[0, "accuracy"] == pytest.approx(0.75)
    assert scored.loc[0, "n"] == 4
    assert scored.loc[0, "ci_low"] <= 0.75 <= scored.loc[0, "ci_high"]


def test_pairwise_mcnemar_counts_discordant_pairs_only():
    """Items both arms get right, and items both get wrong, carry no evidence. Here arm_a
    alone is right on one item and arm_b alone on none, so the discordant count is 1."""
    frames = {
        "arm_a": make_frame(["a_one", "a_two", "X", "Y"], GOLD),
        "arm_b": make_frame(["a_one", "X", "X", "Y"], GOLD),
    }
    out = results.pairwise_mcnemar(frames)
    assert len(out) == 1
    assert out.loc[0, "only_a_correct"] == 1
    assert out.loc[0, "only_b_correct"] == 0
    assert out.loc[0, "n_discordant"] == 1


def test_confusion_pairs_flags_same_family():
    """355/906 labels share a prefix family, so an error inside a family is a different
    failure from one across families and the two must not be collapsed."""
    frame = make_frame(["a_two", "b_one", "b_one", "a_one"], GOLD)
    pairs = results.confusion_pairs(frame)
    flags = dict(zip(zip(pairs["gold"], pairs["pred"]), pairs["same_family"]))
    assert flags[("a_one", "a_two")] is True or flags[("a_one", "a_two")]
    assert not flags[("b_two", "a_one")]


def make_arm3_frame():
    """Three strata: kept rank 1, fell back, moved off rank 1 (one rescue, one breakage)."""
    return pd.DataFrame(
        {
            "question": ["q0", "q1", "q2", "q3"],
            "gold": ["a", "b", "c", "d"],
            "arm1_pred": ["a", "b", "c", "x"],
            "pred": ["a", "b", "z", "d"],
            "parsed_rank": [1.0, None, 3.0, 2.0],
            "fallback_fired": [False, True, False, False],
        }
    )


def test_rerank_decomposition_separates_the_three_strata():
    out = results.rerank_decomposition(make_arm3_frame()).set_index("stratum")
    assert out.loc["kept_rank_1", "n"] == 1
    assert out.loc["fell_back", "n"] == 1
    assert out.loc["moved_off_rank_1", "n"] == 2


def test_rerank_decomposition_fallback_stratum_is_inert():
    """A fallback returns Arm 1's top-1 unchanged, so its accuracy must equal Arm 1's on
    exactly those items — a non-zero delta there would mean the fallback is not inert."""
    out = results.rerank_decomposition(make_arm3_frame()).set_index("stratum")
    assert out.loc["fell_back", "accuracy_delta"] == pytest.approx(0.0)
    assert out.loc["kept_rank_1", "accuracy_delta"] == pytest.approx(0.0)


def test_rerank_decomposition_prices_the_interventions():
    """On the two moved items the LLM breaks one (c -> z) and rescues one (x -> d), so it
    scores the same as Arm 1 overall while having changed both answers."""
    out = results.rerank_decomposition(make_arm3_frame()).set_index("stratum")
    assert out.loc["moved_off_rank_1", "arm3_accuracy"] == pytest.approx(0.5)
    assert out.loc["moved_off_rank_1", "arm1_accuracy"] == pytest.approx(0.5)


def test_assert_reproduces_arm1_test_accepts_matching_shortlist(tmp_path):
    reference = tmp_path / "arm1_test_predictions.csv"
    make_frame(["a_one", "a_two", "b_one", "b_two"], GOLD).to_csv(reference, index=False)
    shortlists = [[label, "filler"] for label in ["a_one", "a_two", "b_one", "b_two"]]
    results.assert_reproduces_arm1_test(shortlists, reference_path=reference)


def test_assert_reproduces_arm1_test_rejects_degraded_index(tmp_path):
    """The 2026-08-23 incident in miniature: a degraded index still yields plausible
    shortlists, so only an item-for-item comparison against the frozen run catches it."""
    reference = tmp_path / "arm1_test_predictions.csv"
    make_frame(["a_one", "a_two", "b_one", "b_two"], GOLD).to_csv(reference, index=False)
    shortlists = [[label, "filler"] for label in ["a_one", "a_two", "b_one", "DIFFERENT"]]
    with pytest.raises(ValueError, match="not the frozen one"):
        results.assert_reproduces_arm1_test(shortlists, reference_path=reference)


def test_assert_reproduces_arm1_test_rejects_length_mismatch(tmp_path):
    reference = tmp_path / "arm1_test_predictions.csv"
    make_frame(["a_one", "a_two", "b_one", "b_two"], GOLD).to_csv(reference, index=False)
    with pytest.raises(ValueError, match="reference holds"):
        results.assert_reproduces_arm1_test([["a_one"]], reference_path=reference)


def test_worked_examples_returns_both_outcomes():
    frames = {"arm_x": make_frame(["a_one", "a_two", "WRONG", "ALSO_WRONG"], GOLD)}
    examples = results.worked_examples(frames, n=2)
    assert set(examples["outcome"]) == {"correct", "incorrect"}
    assert len(examples) == 4


def test_worked_examples_is_seed_stable():
    """The report quotes these examples, so they must not change between reruns."""
    frames = {"arm_x": make_frame(["a_one", "a_two", "WRONG", "ALSO_WRONG"], GOLD)}
    first = results.worked_examples(frames, n=2)
    second = results.worked_examples(frames, n=2)
    pd.testing.assert_frame_equal(first, second)


def test_load_available_arms_returns_only_what_exists(tmp_path):
    """05_results must run standalone on a machine where the Colab half never ran."""
    (tmp_path / "arm1_test_predictions.csv").write_text("question,gold,pred\nq,a,a\n")
    frames = results.load_available_arms(artefacts_dir=tmp_path)
    assert list(frames) == ["arm1_tfidf_knn"]


# --- Defect A: the within-family error rate needs a conditioned denominator ---

LABEL_SPACE = ["a_one", "a_two", "b_one", "solo", "other"]


def test_prefix_family_sizes_counts_over_the_whole_label_space():
    sizes = results.prefix_family_sizes(LABEL_SPACE)
    assert sizes["a"] == 2
    assert sizes["b"] == 1
    assert sizes["solo"] == 1


def test_family_error_summary_excludes_golds_with_no_sibling():
    """A gold whose prefix family has one member cannot be confused with a sibling, so it
    contributes a guaranteed zero to the numerator and must not inflate the denominator.
    Here only the `a_one` error is a possible family error; the `solo` error is not."""
    frame = make_frame(["a_two", "other"], ["a_one", "solo"])
    out = results.family_error_summary(frame, LABEL_SPACE)
    assert out["n_errors"] == 2
    assert out["n_family_error_possible"] == 1
    assert out["n_within_family"] == 1
    assert out["within_family_share"] == pytest.approx(1.0)


def test_family_error_summary_returns_none_share_when_nothing_was_possible():
    """The defect this guards: an unconditioned rate reads 0% when no error *could* have
    been within-family, which is a fact about the sample's class composition, not the
    model. None forces the caller to notice the absent denominator."""
    frame = make_frame(["other"], ["solo"])
    out = results.family_error_summary(frame, LABEL_SPACE)
    assert out["n_family_error_possible"] == 0
    assert out["within_family_share"] is None


def test_family_error_summary_ignores_correct_predictions():
    frame = make_frame(["a_one", "a_two"], ["a_one", "a_one"])
    assert results.family_error_summary(frame, LABEL_SPACE)["n_errors"] == 1


def test_confusion_pairs_flags_whether_a_family_error_was_possible():
    frame = make_frame(["a_two", "other"], ["a_one", "solo"])
    pairs = results.confusion_pairs(frame, label_space=LABEL_SPACE)
    possible = dict(zip(pairs["gold"], pairs["family_error_possible"]))
    assert possible["a_one"]
    assert not possible["solo"]


def test_confusion_pairs_omits_the_flag_without_a_label_space():
    """Back-compatible: callers that never pass a label space keep the original columns."""
    frame = make_frame(["a_two"], ["a_one"])
    assert "family_error_possible" not in results.confusion_pairs(frame).columns


# --- Defect B: macro-F1 must not depend on a ranking being present ---

def test_score_arms_reports_macro_f1_without_ranking_columns():
    """Arm 3 emits a single label, not an ordering. macro-F1 needs only top-1, so gating it
    on the top_* columns silently dropped it for the one arm the error analysis most needs."""
    frames = {"ranking_less": pd.DataFrame({"question": ["q1", "q2"], "gold": ["a_one", "a_two"],
                                            "pred": ["a_one", "b_one"]})}
    scored = results.score_arms(frames)
    assert scored.loc[0, "macro_f1"] == pytest.approx(
        f1_score(["a_one", "a_two"], ["a_one", "b_one"], average="macro", zero_division=0)
    )


def test_score_arms_marks_ranked_metrics_unavailable_rather_than_absent():
    """A blank cell must read as 'cannot produce a ranking', not as a value gone missing."""
    frames = {"ranking_less": pd.DataFrame({"question": ["q1"], "gold": ["a_one"], "pred": ["a_one"]})}
    scored = results.score_arms(frames)
    assert "acc_at_5" in scored.columns and "mrr" in scored.columns
    assert scored.loc[0, "acc_at_5"] is None or pd.isna(scored.loc[0, "acc_at_5"])
