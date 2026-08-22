"""Arm 3 LLM shortlist selection.

Frozen Arm 1 retrieval produces a shortlist of candidate labels; Arm 3 then
formats that shortlist into one of three prompt conditions, sends the prompt to
a seq2seq LLM, parses the model's answer back to a shortlist index, and falls
back to the Arm 1 top-1 label when parsing fails.

All heavy lifting stays here so the notebook can remain a thin orchestration
layer. Tests can inject a fake generator and never download a model.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd

from amlh import arm1_experiments, evaluate
from amlh.config import HYPERPARAMETERS, SEED

PromptMode = Literal["zero_shot", "few_shot", "cot"]


@dataclass(frozen=True)
class PromptExample:
    question: str
    label: str


def prettify_label(label: str) -> str:
    """Render label text in prompt-friendly form without changing its identity."""
    return label.replace("_", " ")


def selected_model_name(hp=HYPERPARAMETERS) -> str:
    return hp.arm3_model_name or "google/flan-t5-large"


def shortlist_k(hp=HYPERPARAMETERS) -> int:
    if hp.shortlist_k is None:
        raise ValueError("shortlist_k must be frozen in config.py before Arm 3 runs")
    return hp.shortlist_k


def n_shots(hp=HYPERPARAMETERS) -> int:
    if hp.n_shots is None:
        return 0
    return hp.n_shots


def llm_temperature(hp=HYPERPARAMETERS) -> float:
    if hp.llm_temperature is None:
        raise ValueError("llm_temperature must be frozen in config.py before Arm 3 runs")
    return hp.llm_temperature


def vec_kwargs(hp=HYPERPARAMETERS) -> dict:
    return {
        "ngram_range": hp.ngram_range,
        "sublinear_tf": hp.sublinear_tf,
        "min_df": hp.min_df,
        "stop_words": hp.stop_words,
    }


def build_shortlist_ranking(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hp=HYPERPARAMETERS,
    depth: int | None = None,
) -> tuple[list[list[str]], list[float]]:
    """Frozen Arm 1 shortlist ranking for Arm 3.

    The shortlist is produced by the frozen Arm 1 retrieval configuration so
    Arm 3 never re-tunes the retriever it consumes.
    """
    return arm1_experiments.frozen_ranking(fit_df, val_df, hp, depth=depth)


def build_examples(fit_df: pd.DataFrame, n: int, seed: int = SEED) -> list[PromptExample]:
    """Deterministically sample compact few-shot exemplars from the fit split only."""
    if n <= 0:
        return []
    sample = fit_df.sample(n=min(n, len(fit_df)), random_state=seed).reset_index(drop=True)
    return [PromptExample(question=row.question, label=row.disease) for row in sample.itertuples()]


def candidate_lines(shortlist: list[str]) -> list[str]:
    return [f"{i + 1}. {prettify_label(label)}" for i, label in enumerate(shortlist)]


def build_prompt(
    question: str,
    shortlist: list[str],
    mode: PromptMode,
    examples: list[PromptExample] | None = None,
) -> str:
    """Build one inspectable prompt string for the requested condition."""
    examples = examples or []
    lines: list[str] = []
    lines.append("You are classifying a patient question into one of the candidate disease labels.")
    lines.append("Choose the single best candidate by number.")
    if mode == "zero_shot":
        lines.append("Respond with only the candidate number.")
    elif mode == "few_shot":
        lines.append("Use the worked examples first, then answer with only the candidate number.")
    elif mode == "cot":
        lines.append("Think briefly, then give a short final answer as 'Final answer: N'.")
    else:
        raise ValueError(f"unknown prompt mode: {mode!r}")

    if examples:
        lines.append("")
        lines.append("Worked examples:")
        for i, example in enumerate(examples, start=1):
            lines.append(f"Example {i}:")
            lines.append(f"Question: {example.question}")
            lines.append(f"Correct label: {prettify_label(example.label)}")
            lines.append("")

    lines.append(f"Question: {question}")
    lines.append("Candidates:")
    lines.extend(candidate_lines(shortlist))
    return "\n".join(lines).strip()


def build_prompts_for_condition(
    val_df: pd.DataFrame,
    shortlist_rankings: list[list[str]],
    mode: PromptMode,
    examples: list[PromptExample] | None = None,
) -> list[str]:
    """Return one prompt per validation question, aligned to `val_df`."""
    prompts = []
    for question, shortlist in zip(val_df["question"].tolist(), shortlist_rankings):
        prompts.append(build_prompt(question, shortlist, mode, examples=examples))
    return prompts


def prompt_token_lengths(prompts: list[str], tokeniser, max_length: int = 512) -> dict:
    """Measure prompt lengths and the truncation rate at the model encoder limit."""
    lengths = [len(tokeniser(prompt, truncation=False)["input_ids"]) for prompt in prompts]
    trunc_rate = sum(length > max_length for length in lengths) / len(lengths) if lengths else 0.0
    series = pd.Series(lengths, dtype="int64") if lengths else pd.Series(dtype="int64")
    return {
        "lengths": lengths,
        "truncation_rate": trunc_rate,
        "min": int(series.min()) if lengths else 0,
        "median": float(series.median()) if lengths else 0.0,
        "mean": float(series.mean()) if lengths else 0.0,
        "p95": float(series.quantile(0.95)) if lengths else 0.0,
        "max": int(series.max()) if lengths else 0,
        "n": len(lengths),
    }


_INDEX_RE = re.compile(r"(?:final\s*answer\s*[:=]?\s*)?(\d+)", re.IGNORECASE)


def parse_candidate_index(raw_output: str, n_candidates: int) -> int | None:
    """Parse a 1-based candidate index from the model output.

    The parser is intentionally strict: if the output cannot be mapped to a
    valid shortlist position, the caller falls back to Arm 1's top-1 label.
    """
    if not raw_output:
        return None
    match = _INDEX_RE.search(raw_output.strip())
    if match is None:
        return None
    idx = int(match.group(1))
    if 1 <= idx <= n_candidates:
        return idx
    return None


def parse_model_output(raw_output: str, shortlist: list[str]) -> tuple[str, bool, int | None]:
    """Map a raw generation to a shortlist label, or fall back to top-1."""
    parsed_index = parse_candidate_index(raw_output, len(shortlist))
    if parsed_index is None:
        return shortlist[0], True, None
    return shortlist[parsed_index - 1], False, parsed_index


def load_generator(model_name: str | None = None, device=None):
    """Load Flan-T5 in float32 for greedy seq2seq generation."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = model_name or selected_model_name()
    tokeniser = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.float32)
    if device is not None:
        model.to(device)
    model.eval()
    return tokeniser, model


def generate_raw_output(
    prompt: str,
    tokeniser,
    model,
    device,
    max_new_tokens: int = 8,
) -> str:
    """Greedy seq2seq generation for one prompt."""
    import torch

    encoded = tokeniser(prompt, return_tensors="pt", truncation=True)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )
    return tokeniser.decode(output_ids[0], skip_special_tokens=True).strip()


def run_condition(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hp=HYPERPARAMETERS,
    mode: PromptMode = "zero_shot",
    tokeniser=None,
    model=None,
    device=None,
    generator: Callable[[str], str] | None = None,
    examples: list[PromptExample] | None = None,
    shortlist_depth: int | None = None,
) -> tuple[pd.DataFrame, dict, list[str]]:
    """Run one prompt condition over the validation split.

    Returns the per-item frame, a metrics dict, and the prompt strings used.
    The caller decides whether to persist them.
    """
    shortlist_depth = shortlist_depth or shortlist_k(hp)
    ranked, top_sim = build_shortlist_ranking(fit_df, val_df, hp, depth=shortlist_depth)
    prompts = build_prompts_for_condition(val_df, ranked, mode, examples=examples)

    rows = []
    start = time.perf_counter()
    for item_idx, (question, gold, shortlist, prompt, sim) in enumerate(
        zip(val_df["question"], val_df["disease"], ranked, prompts, top_sim)
    ):
        if generator is not None:
            raw_output = generator(prompt)
        else:
            if tokeniser is None or model is None or device is None:
                raise ValueError("tokeniser, model and device are required when generator is not supplied")
            raw_output = generate_raw_output(prompt, tokeniser, model, device)

        pred, fallback_fired, parsed_index = parse_model_output(raw_output, shortlist)
        gold_rank = shortlist.index(gold) + 1 if gold in shortlist else None
        rows.append(
            {
                "condition": mode,
                "item_idx": item_idx,
                "question": question,
                "gold": gold,
                "arm1_pred": shortlist[0],
                "pred": pred,
                "parsed_index": parsed_index,
                "fallback_fired": fallback_fired,
                "gold_rank": gold_rank,
                "top_sim": sim,
                "raw_output": raw_output,
                "prompt": prompt,
            }
        )

    pred_df = pd.DataFrame(rows)
    elapsed = time.perf_counter() - start
    metrics = summarise_condition(pred_df, elapsed)
    return pred_df, metrics, prompts


def summarise_condition(pred_df: pd.DataFrame, wall_clock_sec: float) -> dict:
    """Condition-level summary with accuracy, fallback rate, CI and wall clock."""
    pred = pred_df["pred"].tolist()
    gold = pred_df["gold"].tolist()
    ci = evaluate.bootstrap_accuracy_ci(pred, gold, seed=SEED)
    accuracy = sum(p == g for p, g in zip(pred, gold)) / len(gold)
    fallback_rate = float(pred_df["fallback_fired"].mean())
    return {
        "condition": pred_df["condition"].iloc[0] if len(pred_df) else None,
        "n": len(pred_df),
        "accuracy": accuracy,
        "fallback_rate": fallback_rate,
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "wall_clock_sec": wall_clock_sec,
    }


def pairwise_condition_mcnemar(condition_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairwise McNemar comparisons over matched validation items."""
    rows = []
    modes = sorted(condition_frames)
    for i, a in enumerate(modes):
        for b in modes[i + 1 :]:
            frame_a = condition_frames[a]
            frame_b = condition_frames[b]
            result = evaluate.mcnemar_exact(frame_a["pred"].tolist(), frame_b["pred"].tolist(), frame_a["gold"].tolist())
            rows.append({"condition_a": a, "condition_b": b, **result})
    return pd.DataFrame(rows)


def select_prompt_mode(condition_metrics: pd.DataFrame, mcnemar_df: pd.DataFrame) -> tuple[str, bool]:
    """Select the prompt mode, returning (mode, tie_break_fired).

    If no condition is significantly better than both others, the unresolved
    branch keeps `zero_shot` on the declared prior.
    """
    wins: dict[str, set[str]] = {mode: set() for mode in condition_metrics["condition"]}
    for row in mcnemar_df.itertuples(index=False):
        if row.p_value >= 0.05:
            continue
        if row.accuracy_a > row.accuracy_b:
            wins[row.condition_a].add(row.condition_b)
        elif row.accuracy_b > row.accuracy_a:
            wins[row.condition_b].add(row.condition_a)

    dominating = [mode for mode, beaten in wins.items() if len(beaten) == len(condition_metrics) - 1]
    if len(dominating) == 1:
        return dominating[0], False
    return "zero_shot", True


def build_prompt_table(prompts_by_mode: dict[str, list[str]]) -> str:
    """Concatenate prompt strings for persistence in `arm3_prompts.txt`."""
    sections = []
    for mode, prompts in prompts_by_mode.items():
        sections.append(f"### {mode}")
        sections.extend(prompts)
        sections.append("")
    return "\n".join(sections).strip() + "\n"
