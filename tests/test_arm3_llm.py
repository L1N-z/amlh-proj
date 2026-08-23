import pandas as pd
import pytest

from amlh import arm3_llm as a3


class FakeTokenizer:
    def __call__(self, text, truncation=False, return_tensors=None):
        tokens = text.split()
        payload = {"input_ids": list(range(len(tokens))) if not return_tensors else [list(range(len(tokens)))]}
        return payload


def fake_pipe(outputs):
    """A `pipe` with the practical's return shape, replaying `outputs` in order."""
    stream = iter(outputs)
    calls = []

    def pipe(prompt, max_new_tokens=100, temperature=0.0, **kwargs):
        calls.append({"prompt": prompt, "max_new_tokens": max_new_tokens})
        return [{"generated_text": next(stream)}]

    pipe.calls = calls
    return pipe


@pytest.fixture
def tiny_fit():
    return pd.DataFrame(
        {
            "question": ["how do I stop cough", "why do I feel fever", "what causes rash"],
            "disease": ["bronchitis", "flu", "eczema"],
        }
    )


@pytest.fixture
def tiny_val():
    return pd.DataFrame(
        {
            "question": ["why do I cough", "what causes rash"],
            "disease": ["bronchitis", "eczema"],
        }
    )


@pytest.fixture
def tiny_rankings():
    return [["bronchitis", "flu"], ["eczema", "flu"]]


def test_build_prompt_is_chat_messages_with_practical_phrasing():
    messages = a3.build_prompt("why do I cough", ["chronic_cough", "flu_like_illness"], "zero_shot")

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == a3.SYSTEM_PROMPT
    user = messages[1]["content"]
    assert "which one of the following diagnoses is most relevant?" in user
    assert "Respond with only the diagnosis name." in user
    assert "chronic cough, flu like illness" in user
    assert "_" not in user


def test_build_prompt_cot_asks_for_reasoning_without_a_copyable_placeholder():
    user = a3.build_prompt("why do I cough", ["chronic_cough"], "cot")[1]["content"]

    assert "Think step by step" in user
    assert "'Final answer:'" in user
    # The first run's prompt said "'Final answer: N'" and the model copied the N.
    assert "Final answer: N" not in user


def test_build_prompt_few_shot_includes_compact_examples():
    user = a3.build_prompt(
        "why do I cough",
        ["chronic_cough", "flu_like_illness"],
        "few_shot",
        examples=[a3.PromptExample(question="I cough", label="chronic_cough")],
    )[1]["content"]

    assert "Worked examples:" in user
    assert "Diagnosis:\nchronic cough" in user


def test_build_prompt_rejects_unknown_mode():
    with pytest.raises(ValueError):
        a3.build_prompt("q", ["a"], "self_consistency")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("chronic cough", "chronic_cough"),                              # exact, as the practical matches
        ("Chronic Cough.", "chronic_cough"),                             # punctuation and case
        ("The most relevant diagnosis is chronic cough.", "chronic_cough"),  # wrapped in a sentence
        ("Final answer: flu like illness", "flu_like_illness"),          # cot marker
        ("Let me think. Maybe flu.\nFinal answer: chronic cough", "chronic_cough"),  # last marker wins
        ("N", None),                                                     # the placeholder the old prompt leaked
        ("Final answer: N", None),
        ("", None),
        ("something entirely unrelated", None),
    ],
)
def test_parse_diagnosis_name(raw, expected):
    assert a3.parse_diagnosis_name(raw, ["chronic_cough", "flu_like_illness"]) == expected


def test_parse_diagnosis_name_prefers_the_longest_matching_candidate():
    shortlist = ["leukaemia", "acute_myeloid_leukaemia"]
    assert a3.parse_diagnosis_name("acute myeloid leukaemia", shortlist) == "acute_myeloid_leukaemia"


def test_parse_model_output_reports_rank_and_falls_back_to_arm1_top1():
    shortlist = ["alpha", "beta"]

    label, fallback, matched, rank = a3.parse_model_output("beta", shortlist)
    assert (label, fallback, matched, rank) == ("beta", False, "beta", 2)

    label, fallback, matched, rank = a3.parse_model_output("no answer", shortlist)
    assert (label, fallback, matched, rank) == ("alpha", True, None, None)


def test_max_new_tokens_gives_cot_room_to_reason():
    assert a3.max_new_tokens_for("cot") > a3.max_new_tokens_for("zero_shot")
    assert a3.max_new_tokens_for("zero_shot") == 20  # the practical's budget


def test_prompt_token_lengths_reports_truncation_rate():
    prompts = ["one two three", "one two three four five"]
    summary = a3.prompt_token_lengths(prompts, FakeTokenizer(), max_length=4)
    assert summary["n"] == 2
    assert summary["lengths"] == [3, 5]
    assert summary["truncation_rate"] == pytest.approx(0.5)


def test_prompt_token_lengths_accepts_chat_messages():
    prompts = [a3.build_prompt("why do I cough", ["chronic_cough"], "zero_shot")]
    summary = a3.prompt_token_lengths(prompts, FakeTokenizer(), max_length=4)
    assert summary["n"] == 1
    assert summary["lengths"][0] > 4


def test_run_condition_records_predictions_and_fallbacks(tiny_fit, tiny_val, tiny_rankings):
    pipe = fake_pipe(["flu", "not a diagnosis at all"])

    pred_df, metrics, prompts = a3.run_condition(
        tiny_fit,
        tiny_val,
        mode="zero_shot",
        pipe=pipe,
        shortlist_rankings=tiny_rankings,
        top_sim=[0.9, 0.8],
        model_name="fake/model",
    )

    assert len(prompts) == len(tiny_val)
    assert list(pred_df["pred"]) == ["flu", "eczema"]
    assert list(pred_df["fallback_fired"]) == [False, True]
    assert list(pred_df["arm1_pred"]) == ["bronchitis", "eczema"]
    assert pred_df["parsed_rank"].iloc[0] == 2
    assert pd.isna(pred_df["parsed_rank"].iloc[1])  # no rank recorded when the fallback fires
    assert set(pred_df["model_name"]) == {"fake/model"}
    assert metrics["n"] == 2
    assert metrics["fallback_rate"] == pytest.approx(0.5)


def test_run_condition_reports_the_arm1_baseline_it_was_handed(tiny_fit, tiny_val, tiny_rankings):
    pipe = fake_pipe(["flu", "flu"])

    _, metrics, _ = a3.run_condition(
        tiny_fit, tiny_val, mode="zero_shot", pipe=pipe, shortlist_rankings=tiny_rankings, top_sim=[0.9, 0.8]
    )

    # The LLM moved both items off rank 1 and got both wrong; Arm 1 had both right.
    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["arm1_accuracy"] == pytest.approx(1.0)
    assert metrics["accuracy_minus_arm1"] == pytest.approx(-1.0)
    assert metrics["moved_off_rank1_rate"] == pytest.approx(1.0)


def test_run_condition_passes_the_per_mode_token_budget(tiny_fit, tiny_val, tiny_rankings):
    pipe = fake_pipe(["flu", "flu"])
    a3.run_condition(
        tiny_fit, tiny_val, mode="cot", pipe=pipe, shortlist_rankings=tiny_rankings, top_sim=[0.9, 0.8]
    )
    assert {call["max_new_tokens"] for call in pipe.calls} == {a3.max_new_tokens_for("cot")}


def test_run_condition_requires_a_pipe(tiny_fit, tiny_val):
    with pytest.raises(ValueError, match="pipe"):
        a3.run_condition(tiny_fit, tiny_val, mode="zero_shot", pipe=None)


def test_assert_reproduces_arm1_accepts_a_matching_shortlist(tiny_rankings):
    arm1 = pd.DataFrame({"top_1": ["bronchitis", "eczema"]})
    assert a3.assert_reproduces_arm1(tiny_rankings, arm1)["top1_matches_arm1"] is True


def test_assert_reproduces_arm1_catches_a_degraded_index_variant(tiny_rankings):
    """The QLAD -> QLA degradation that invalidated the first Arm 3 run."""
    arm1 = pd.DataFrame({"top_1": ["bronchitis", "flu"]})
    with pytest.raises(ValueError, match="disagrees with Arm 1 on 1/2"):
        a3.assert_reproduces_arm1(tiny_rankings, arm1)


def test_assert_reproduces_arm1_catches_a_length_mismatch(tiny_rankings):
    with pytest.raises(ValueError, match="same validation split"):
        a3.assert_reproduces_arm1(tiny_rankings, pd.DataFrame({"top_1": ["bronchitis"]}))


def test_pairwise_condition_mcnemar_and_selection():
    base = pd.DataFrame({"gold": ["a", "a", "b", "b"], "pred": ["a", "x", "b", "x"]})
    alt = pd.DataFrame({"gold": ["a", "a", "b", "b"], "pred": ["a", "a", "x", "x"]})
    other = pd.DataFrame({"gold": ["a", "a", "b", "b"], "pred": ["x", "x", "b", "x"]})
    frames = {
        "zero_shot": base.assign(condition="zero_shot"),
        "few_shot": alt.assign(condition="few_shot"),
        "cot": other.assign(condition="cot"),
    }
    mcnemar = a3.pairwise_condition_mcnemar(frames)
    assert set(zip(mcnemar["condition_a"], mcnemar["condition_b"])) == {
        ("cot", "few_shot"),
        ("cot", "zero_shot"),
        ("few_shot", "zero_shot"),
    }

    metrics = pd.DataFrame({"condition": ["zero_shot", "few_shot", "cot"], "accuracy": [0.5, 0.75, 0.25]})
    chosen, tie_break = a3.select_prompt_mode(metrics, mcnemar)
    assert chosen == "zero_shot"
    assert tie_break is True


def test_condition_vs_arm1_mcnemar_pairs_each_condition_with_the_retriever():
    frame = pd.DataFrame(
        {
            "condition": ["zero_shot"] * 4,
            "gold": ["a", "a", "b", "b"],
            "pred": ["a", "x", "x", "x"],
            "arm1_pred": ["a", "a", "b", "x"],
        }
    )
    result = a3.condition_vs_arm1_mcnemar({"zero_shot": frame})
    row = result.iloc[0]
    assert row.system_a == "arm3_zero_shot"
    assert row.system_b == "arm1_top1"
    assert row.only_b_correct == 2  # arm1 right, llm wrong on items 2 and 3
    assert row.only_a_correct == 0


def test_select_model_keeps_the_in_domain_prior_when_mcnemar_is_unresolved():
    unresolved = pd.DataFrame(
        [{"system_a": "primary", "system_b": "secondary", "accuracy_a": 0.4, "accuracy_b": 0.6, "p_value": 0.5}]
    )
    chosen, tie_break = a3.select_model(unresolved)
    assert chosen == a3.selected_model_name()
    assert tie_break is True


def test_select_model_lets_a_resolved_comparison_decide():
    resolved = pd.DataFrame(
        [{"system_a": "primary", "system_b": "secondary", "accuracy_a": 0.4, "accuracy_b": 0.6, "p_value": 0.01}]
    )
    chosen, tie_break = a3.select_model(resolved)
    assert chosen == "secondary"
    assert tie_break is False


def test_build_prompt_table_contains_sections():
    text = a3.build_prompt_table({"zero_shot": ["p1"], "few_shot": ["p2", "p3"]})
    assert "### zero_shot" in text
    assert "### few_shot" in text
    assert text.endswith("\n")


def test_build_prompt_table_renders_chat_messages():
    prompts = [a3.build_prompt("why do I cough", ["chronic_cough"], "zero_shot")]
    text = a3.build_prompt_table({"zero_shot": prompts})
    assert a3.SYSTEM_PROMPT in text
    assert "Respond with only the diagnosis name." in text
