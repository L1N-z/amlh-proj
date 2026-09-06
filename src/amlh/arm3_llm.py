"""Arm 3: LLM selection over an Arm 1 shortlist.

Frozen Arm 1 retrieval produces a shortlist of candidate labels; Arm 3 formats
it into one of three prompt conditions, sends the prompt to the generator,
matches the reply back to a shortlist label by name, and falls back to the Arm 1
top-1 label when no candidate matches.

The design follows Task 4 of the NLP4 practical, which prompts an LLM to pick the
most relevant diagnosis from a candidate list: the generator is called through a
``pipe(prompt, max_new_tokens=...)`` closure, prompts are chat messages carrying a
system persona and a "Respond with only the diagnosis name." instruction, and the
reply is matched by name rather than by a candidate number.

Two deviations from the practical, both stated in the report:

1. The practical falls back to a random diagnosis when the reply matches nothing.
   Arm 3 returns the Arm 1 top-1 instead — the prediction the system would have
   made without the LLM — so the fallback adds no noise of its own.
2. ``flatten_messages`` joins roles with a blank line where the practical's API
   branch joins them directly.

All heavy lifting stays here so the notebook remains a thin orchestration layer;
tests can inject a fake generator and never download a model.
"""

from __future__ import annotations

import contextlib
import re
import time
from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd

from amlh import arm1_experiments, evaluate
from amlh.config import HYPERPARAMETERS, SEED

PromptMode = Literal["zero_shot", "few_shot", "cot"]

# Verbatim from the practical's diagnosis-selection prompt (cell-27).
SYSTEM_PROMPT = "You are a helpful assistant trained to identify relevant medical question topics."

# Per-condition closing instruction. `cot` carries no placeholder token for the
# model to copy: the first Arm 3 run asked for "'Final answer: N'" and
# Flan-T5-large echoed the literal letter N on 93/200 items, so 46.5% of that
# condition was fallback rather than model choice.
_CLOSERS: dict[str, str] = {
    "zero_shot": "Respond with only the diagnosis name.",
    "few_shot": "Respond with only the diagnosis name.",
    "cot": (
        "Think step by step about which diagnosis the question is asking about, "
        "then end your reply with 'Final answer:' followed by the diagnosis name."
    ),
}


@dataclass(frozen=True)
class PromptExample:
    question: str
    label: str


def prettify_label(label: str) -> str:
    """Render label text in prompt-friendly form without changing its identity."""
    return label.replace("_", " ")


def selected_model_name(hp=HYPERPARAMETERS) -> str:
    return hp.arm3_model_name or "microsoft/MediPhi-Guidelines"


def secondary_model_name(hp=HYPERPARAMETERS) -> str:
    return hp.arm3_secondary_model_name or "google/flan-t5-large"


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


def max_new_tokens_for(mode: PromptMode, hp=HYPERPARAMETERS) -> int:
    """Generation budget for `mode`.

    `cot` needs room for reasoning *and* a final answer; the other two need only
    a diagnosis name, for which the practical budgets 20 tokens.
    """
    if mode == "cot":
        return hp.arm3_cot_max_new_tokens or 128
    return hp.arm3_max_new_tokens or 20


def build_shortlist_ranking(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hp=HYPERPARAMETERS,
    depth: int | None = None,
) -> tuple[list[list[str]], list[float]]:
    """Frozen Arm 1 shortlist ranking for Arm 3.

    The shortlist is produced by the frozen Arm 1 retrieval configuration so
    Arm 3 never re-tunes the retriever it consumes. `features.build_index`
    raises if the NHS documents backing the frozen QLAD variant are absent, so
    this cannot quietly return a lower-variant ranking.
    """
    return arm1_experiments.frozen_ranking(fit_df, val_df, hp, depth=depth)


def assert_reproduces_arm1(
    shortlist_rankings: list[list[str]], arm1_predictions: pd.DataFrame
) -> dict:
    """Check the shortlist's top-1 against Arm 1's persisted per-item predictions.

    Rank 1 identifies the index that produced it, so a shortlist built from
    anything other than the frozen configuration is caught here, before any
    prompt is sent.
    """
    top1 = [ranking[0] for ranking in shortlist_rankings]
    expected = arm1_predictions["top_1"].tolist()
    if len(top1) != len(expected):
        raise ValueError(
            f"shortlist has {len(top1)} items but arm1_val_predictions.csv has {len(expected)}; "
            "these must be the same validation split in the same order"
        )
    mismatches = [i for i, (a, b) in enumerate(zip(top1, expected)) if a != b]
    if mismatches:
        raise ValueError(
            f"shortlist top-1 disagrees with Arm 1 on {len(mismatches)}/{len(expected)} items "
            f"(first at index {mismatches[0]}: got {top1[mismatches[0]]!r}, "
            f"expected {expected[mismatches[0]]!r}). The frozen retrieval configuration is not "
            "the one that produced arm1_val_predictions.csv — check that data/db_nhs_qa_classification "
            "is present and that config.py has not drifted."
        )
    return {"n": len(top1), "top1_matches_arm1": True}


def build_examples(fit_df: pd.DataFrame, n: int, seed: int = SEED) -> list[PromptExample]:
    """Deterministically sample compact few-shot exemplars from the fit split only."""
    if n <= 0:
        return []
    sample = fit_df.sample(n=min(n, len(fit_df)), random_state=seed).reset_index(drop=True)
    return [PromptExample(question=row.question, label=row.disease) for row in sample.itertuples()]


def candidate_names(shortlist: list[str]) -> str:
    """Candidate labels as the practical renders them: a comma-joined name list."""
    return ", ".join(prettify_label(label) for label in shortlist)


def build_prompt(
    question: str,
    shortlist: list[str],
    mode: PromptMode,
    examples: list[PromptExample] | None = None,
) -> list[dict[str, str]]:
    """Build one chat-message prompt for the requested condition.

    Follows the practical's diagnosis-selection prompt, with its candidate list
    narrowed from every disease to the Arm 1 shortlist.
    """
    if mode not in _CLOSERS:
        raise ValueError(f"unknown prompt mode: {mode!r}")
    examples = examples or []

    parts = [
        "Given the question below, which one of the following diagnoses is most relevant?",
        "",
        "Diagnoses:",
        candidate_names(shortlist),
        "",
    ]

    if examples:
        parts.append("Worked examples:")
        for example in examples:
            parts.append(f"Question:\n{example.question}")
            parts.append(f"Diagnosis:\n{prettify_label(example.label)}")
            parts.append("")

    parts.append(f"Question:\n{question}")
    parts.append("")
    parts.append(_CLOSERS[mode])

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts).strip()},
    ]


def flatten_messages(messages: list[dict[str, str]]) -> str:
    """Chat messages as one string, for tokenisers without a chat template.

    The practical's API branch does ``''.join(item['content'] for item in prompt)``;
    this joins on a blank line instead so the persona does not run into the
    instruction mid-sentence.
    """
    return "\n\n".join(message["content"] for message in messages)


def build_prompts_for_condition(
    val_df: pd.DataFrame,
    shortlist_rankings: list[list[str]],
    mode: PromptMode,
    examples: list[PromptExample] | None = None,
) -> list[list[dict[str, str]]]:
    """Return one chat prompt per validation question, aligned to `val_df`."""
    prompts = []
    for question, shortlist in zip(val_df["question"].tolist(), shortlist_rankings):
        prompts.append(build_prompt(question, shortlist, mode, examples=examples))
    return prompts


def prompt_token_lengths(prompts: list, tokeniser, max_length: int = 512) -> dict:
    """Measure prompt lengths and the truncation rate at the model encoder limit."""
    texts = [flatten_messages(p) if isinstance(p, list) else p for p in prompts]
    lengths = [len(tokeniser(text, truncation=False)["input_ids"]) for text in texts]
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


# Matched per line (no DOTALL) so the *last* marker in a chain of reasoning wins.
_FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*[:\-]?\s*(.*)", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalise(text: str) -> str:
    """Casefold and reduce to single-spaced alphanumerics for name matching."""
    return _NON_ALNUM_RE.sub(" ", text.lower()).strip()


def parse_diagnosis_name(raw_output: str, shortlist: list[str]) -> str | None:
    """Match a generated diagnosis name back to a shortlist label.

    Exact normalised equality first, as in the practical's ``in unique_diagnoses``
    membership test. Instruction-tuned models often wrap the name in a sentence,
    so a containment pass follows; the longest matching candidate wins, otherwise
    "leukaemia" would shadow "acute myeloid leukaemia" on a shortlist holding both.
    Returns None when nothing matches, and the caller falls back to Arm 1's top-1.
    """
    if not raw_output:
        return None
    text = raw_output.strip()
    markers = list(_FINAL_ANSWER_RE.finditer(text))
    if markers:
        text = markers[-1].group(1)

    normalised_output = _normalise(text)
    if not normalised_output:
        return None

    candidates = [(label, _normalise(prettify_label(label))) for label in shortlist]
    for label, normalised_label in candidates:
        if normalised_label == normalised_output:
            return label

    best_label, best_length = None, 0
    for label, normalised_label in candidates:
        if normalised_label and normalised_label in normalised_output and len(normalised_label) > best_length:
            best_label, best_length = label, len(normalised_label)
    return best_label


def parse_model_output(raw_output: str, shortlist: list[str]) -> tuple[str, bool, str | None, int | None]:
    """Map a raw generation to a shortlist label, or fall back to top-1.

    Returns ``(label, fallback_fired, matched_label, matched_rank)``, where
    `matched_rank` is the 1-based shortlist position the model moved to — the
    quantity that says whether the LLM is reordering the retriever at all.
    """
    matched = parse_diagnosis_name(raw_output, shortlist)
    if matched is None:
        return shortlist[0], True, None, None
    return matched, False, matched, shortlist.index(matched) + 1


def load_generator(model_name: str | None = None, device=None, hp=HYPERPARAMETERS):
    """Load an Arm 3 generator and return ``(tokenizer, model, pipe)``.

    `pipe` has the practical's signature and return shape
    (``[{"generated_text": ...}]``) so the notebook's call site reads the same.
    Encoder-decoder checkpoints (Flan-T5) have no chat template, so their
    prompts are flattened to text; causal checkpoints (MediPhi) go through
    ``apply_chat_template`` exactly as the practical does.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    name = model_name or selected_model_name(hp)
    config = AutoConfig.from_pretrained(name)
    is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(name)
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    loader = AutoModelForSeq2SeqLM if is_encoder_decoder else AutoModelForCausalLM
    model = loader.from_pretrained(name, dtype=dtype).to(device)
    model.eval()

    def pipe(
        prompt,
        max_new_tokens=100,  # maximum number of new tokens to generate (excluding the input prompt)
        temperature=0.0,  # controls randomness, with do_sample=False generation is deterministic
        return_full_text=False,  # if True, return both the prompt and generated text
        do_sample=False,  # if False, use greedy decoding
        clean_up_tokenization_spaces=False,  # whether to remove tokenisation artefacts
    ):
        # prompt is expected to be [{"role": "system", ...}, {"role": "user", ...}]
        if is_encoder_decoder:
            inputs = tokenizer(flatten_messages(prompt), return_tensors="pt", truncation=True)
        else:
            inputs = tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )

        # An encoder-decoder's output holds only the generated tokens; a causal
        # model's replays the prompt first and has to be sliced off.
        if is_encoder_decoder or return_full_text:
            generated_ids = outputs[0]
        else:
            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]

        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        return [{"generated_text": generated_text.strip()}]

    return tokenizer, model, pipe


@contextlib.contextmanager
def generator_session(model_name: str | None = None, device=None, hp=HYPERPARAMETERS):
    """`load_generator`, scoped to a `with` block that always frees the GPU.

    `run_condition` calls `pipe(...)` once per item with no batching; if any call raises
    (CUDA OOM or otherwise), a bare `del model; torch.cuda.empty_cache()` placed after the
    call never runs, because it sits after the very call it was meant to guard. That leaves
    a multi-GB generator resident on the GPU, so a naive retry of the same cell loads a
    second copy on top of it and can exceed the GPU's memory even though either copy alone
    would have fit. The `finally` below runs on both the success and the exception path.
    """
    import torch

    tokenizer, model, pipe = load_generator(model_name, device=device, hp=hp)
    try:
        yield tokenizer, model, pipe
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_condition(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hp=HYPERPARAMETERS,
    mode: PromptMode = "zero_shot",
    pipe: Callable | None = None,
    examples: list[PromptExample] | None = None,
    shortlist_depth: int | None = None,
    model_name: str | None = None,
    shortlist_rankings: list[list[str]] | None = None,
    top_sim: list[float] | None = None,
) -> tuple[pd.DataFrame, dict, list]:
    """Run one prompt condition over the validation split.

    Returns the per-item frame, a metrics dict, and the chat prompts used. The
    caller decides whether to persist them. Passing `shortlist_rankings` reuses
    a ranking already built rather than re-running retrieval per condition.
    """
    if pipe is None:
        raise ValueError("a `pipe` callable is required — pass load_generator(...)[2] or a fake")
    shortlist_depth = shortlist_depth or shortlist_k(hp)
    if shortlist_rankings is None:
        shortlist_rankings, top_sim = build_shortlist_ranking(fit_df, val_df, hp, depth=shortlist_depth)
    if top_sim is None:
        top_sim = [float("nan")] * len(shortlist_rankings)

    prompts = build_prompts_for_condition(val_df, shortlist_rankings, mode, examples=examples)
    budget = max_new_tokens_for(mode, hp)
    temperature = llm_temperature(hp)

    rows = []
    start = time.perf_counter()
    for item_idx, (question, gold, shortlist, prompt, sim) in enumerate(
        zip(val_df["question"], val_df["disease"], shortlist_rankings, prompts, top_sim)
    ):
        raw_output = pipe(prompt, max_new_tokens=budget, temperature=temperature)[0]["generated_text"]
        pred, fallback_fired, matched_label, matched_rank = parse_model_output(raw_output, shortlist)
        gold_rank = shortlist.index(gold) + 1 if gold in shortlist else None
        rows.append(
            {
                "condition": mode,
                "model_name": model_name or selected_model_name(hp),
                "item_idx": item_idx,
                "question": question,
                "gold": gold,
                "arm1_pred": shortlist[0],
                "pred": pred,
                "parsed_label": matched_label,
                "parsed_rank": matched_rank,
                "fallback_fired": fallback_fired,
                "gold_rank": gold_rank,
                "top_sim": sim,
                "raw_output": raw_output,
                "prompt": flatten_messages(prompt),
            }
        )

    pred_df = pd.DataFrame(rows)
    elapsed = time.perf_counter() - start
    metrics = summarise_condition(pred_df, elapsed)
    return pred_df, metrics, prompts


def summarise_condition(pred_df: pd.DataFrame, wall_clock_sec: float) -> dict:
    """Condition-level summary with accuracy, fallback rate, CI and wall clock.

    `arm1_accuracy` is the shortlist's own top-1 on the same items — the
    baseline the LLM has to beat to have earned its place in the pipeline, and
    the comparison the first run never recorded.
    """
    pred = pred_df["pred"].tolist()
    gold = pred_df["gold"].tolist()
    arm1 = pred_df["arm1_pred"].tolist()
    ci = evaluate.bootstrap_accuracy_ci(pred, gold, seed=SEED)
    accuracy = sum(p == g for p, g in zip(pred, gold)) / len(gold)
    arm1_accuracy = sum(p == g for p, g in zip(arm1, gold)) / len(gold)
    non_fallback = pred_df[~pred_df["fallback_fired"]]
    return {
        "condition": pred_df["condition"].iloc[0] if len(pred_df) else None,
        "model_name": pred_df["model_name"].iloc[0] if len(pred_df) else None,
        "n": len(pred_df),
        "accuracy": accuracy,
        "arm1_accuracy": arm1_accuracy,
        "accuracy_minus_arm1": accuracy - arm1_accuracy,
        "fallback_rate": float(pred_df["fallback_fired"].mean()),
        "agreement_with_arm1": float((pred_df["pred"] == pred_df["arm1_pred"]).mean()),
        "moved_off_rank1_rate": float((non_fallback["parsed_rank"] > 1).mean()) if len(non_fallback) else 0.0,
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


def condition_vs_arm1_mcnemar(condition_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """McNemar of each condition against the Arm 1 top-1 it was handed.

    Paired by construction: the LLM and the retriever answer the same 200 items,
    and the items where the LLM simply kept rank 1 carry no evidence either way.
    """
    rows = []
    for mode in sorted(condition_frames):
        frame = condition_frames[mode]
        result = evaluate.mcnemar_exact(frame["pred"].tolist(), frame["arm1_pred"].tolist(), frame["gold"].tolist())
        rows.append({"system_a": f"arm3_{mode}", "system_b": "arm1_top1", **result})
    return pd.DataFrame(rows)


def select_prompt_mode(condition_metrics: pd.DataFrame, mcnemar_df: pd.DataFrame) -> tuple[str, bool]:
    """Select the prompt mode, returning (mode, tie_break_fired).

    If no condition is significantly better than both others, the comparison is
    reported as unresolved and `zero_shot` is kept as the simplest prompt. This
    can and does retain a lower-scoring condition.
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


def model_mcnemar(primary_frame: pd.DataFrame, secondary_frame: pd.DataFrame) -> pd.DataFrame:
    """McNemar of the two Arm 3 generators at the selected prompt condition."""
    result = evaluate.mcnemar_exact(
        primary_frame["pred"].tolist(), secondary_frame["pred"].tolist(), primary_frame["gold"].tolist()
    )
    return pd.DataFrame(
        [
            {
                "system_a": primary_frame["model_name"].iloc[0],
                "system_b": secondary_frame["model_name"].iloc[0],
                **result,
            }
        ]
    )


def select_model(model_mcnemar_df: pd.DataFrame, hp=HYPERPARAMETERS) -> tuple[str, bool]:
    """Select the Arm 3 generator, returning (model_name, tie_break_fired).

    Mirrors the Arm 2 encoder rule: if McNemar does not resolve the pair at
    p < 0.05 the comparison is reported as unresolved and the in-domain clinical
    model is kept on the declared prior. This can and does retain the
    lower-scoring model.
    """
    row = model_mcnemar_df.iloc[0]
    primary = selected_model_name(hp)
    if row.p_value < 0.05:
        winner = row.system_a if row.accuracy_a > row.accuracy_b else row.system_b
        return winner, False
    return primary, True


def build_prompt_table(prompts_by_mode: dict[str, list]) -> str:
    """Concatenate prompt strings for persistence in `arm3_prompts.txt`."""
    sections = []
    for mode, prompts in prompts_by_mode.items():
        sections.append(f"### {mode}")
        for prompt in prompts:
            sections.append(flatten_messages(prompt) if isinstance(prompt, list) else prompt)
        sections.append("")
    return "\n".join(sections).strip() + "\n"
