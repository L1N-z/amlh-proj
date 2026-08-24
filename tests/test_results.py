import pandas as pd
import pytest

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
