import pandas as pd
import pytest

from amlh import arm3_llm as a3


class FakeTokenizer:
    def __call__(self, text, truncation=False, return_tensors=None):
        tokens = text.split()
        payload = {"input_ids": list(range(len(tokens))) if not return_tensors else [list(range(len(tokens)))]}
        return payload


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


def test_build_prompt_numbers_candidates_and_prettifies_labels():
    prompt = a3.build_prompt(
        "why do I cough",
        ["chronic_cough", "flu_like_illness"],
        "zero_shot",
    )
    assert "1. chronic cough" in prompt
    assert "2. flu like illness" in prompt
    assert "_" not in prompt
    assert "Respond with only the candidate number." in prompt


def test_build_prompt_few_shot_includes_compact_examples():
    prompt = a3.build_prompt(
        "why do I cough",
        ["chronic_cough", "flu_like_illness"],
        "few_shot",
        examples=[a3.PromptExample(question="I cough", label="chronic_cough")],
    )
    assert "Worked examples:" in prompt
    assert "Correct label: chronic cough" in prompt
    assert "Candidates:" in prompt


def test_parse_candidate_index_and_fallback():
    assert a3.parse_candidate_index("Final answer: 2", 3) == 2
    assert a3.parse_candidate_index("2", 3) == 2
    assert a3.parse_candidate_index("Final answer: 9", 3) is None
    label, fallback, parsed = a3.parse_model_output("no answer", ["alpha", "beta"])
    assert label == "alpha"
    assert fallback is True
    assert parsed is None


def test_prompt_token_lengths_reports_truncation_rate():
    prompts = ["one two three", "one two three four five"]
    summary = a3.prompt_token_lengths(prompts, FakeTokenizer(), max_length=4)
    assert summary["n"] == 2
    assert summary["lengths"] == [3, 5]
    assert summary["truncation_rate"] == pytest.approx(0.5)


def test_run_condition_records_predictions_and_fallbacks(monkeypatch, tiny_fit, tiny_val):
    monkeypatch.setattr(
        a3,
        "build_shortlist_ranking",
        lambda fit_df, val_df, hp, depth=None: ([
            ["bronchitis", "flu"],
            ["eczema", "flu"],
        ], [0.9, 0.8]),
    )

    outputs = iter(["2", "not parseable"])

    def generator(prompt):
        return next(outputs)

    pred_df, metrics, prompts = a3.run_condition(
        tiny_fit,
        tiny_val,
        mode="zero_shot",
        generator=generator,
        shortlist_depth=2,
    )

    assert len(prompts) == len(tiny_val)
    assert list(pred_df["pred"]) == ["flu", "eczema"]
    assert list(pred_df["fallback_fired"]) == [False, True]
    assert list(pred_df["arm1_pred"]) == ["bronchitis", "eczema"]
    assert metrics["condition"] == "zero_shot"
    assert metrics["n"] == 2
    assert metrics["fallback_rate"] == pytest.approx(0.5)


def test_pairwise_condition_mcnemar_and_selection():
    base = pd.DataFrame(
        {
            "gold": ["a", "a", "b", "b"],
            "pred": ["a", "x", "b", "x"],
        }
    )
    alt = pd.DataFrame(
        {
            "gold": ["a", "a", "b", "b"],
            "pred": ["a", "a", "x", "x"],
        }
    )
    other = pd.DataFrame(
        {
            "gold": ["a", "a", "b", "b"],
            "pred": ["x", "x", "b", "x"],
        }
    )
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

    metrics = pd.DataFrame(
        {
            "condition": ["zero_shot", "few_shot", "cot"],
            "accuracy": [0.5, 0.75, 0.25],
        }
    )
    chosen, tie_break = a3.select_prompt_mode(metrics, mcnemar)
    assert chosen == "zero_shot"
    assert tie_break is True


def test_build_prompt_table_contains_sections():
    text = a3.build_prompt_table({"zero_shot": ["p1"], "few_shot": ["p2", "p3"]})
    assert "### zero_shot" in text
    assert "### few_shot" in text
    assert text.endswith("\n")
