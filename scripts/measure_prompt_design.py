"""Arm 3 prompt-design diagnostics: what each prompt condition actually changes.

Validation only. Reads no test-set quantity, and selects nothing — Arm 3's condition and
generator were frozen on 2026-08-23 by the pre-registered McNemar rules. This is post-hoc
characterisation for the report's prompt-design reflection, which the brief asks for and
which a non-significant p-value on its own does not supply.

Two questions, neither answerable from `arm3_prompt_conditions.csv` as it stands:

1. Accuracy differences between conditions are inside noise (smallest McNemar p = 0.511),
   so what *does* the prompt move? Decomposing each condition into kept-rank-1 / fell-back /
   moved-off-rank-1 shows it moves the override rate, while override precision stays flat —
   the model becomes more willing to depart from the retriever's ranking, not better at it.

2. Fallbacks are recorded as a rate but never characterised. CLAUDE.md records that no
   fallback output matches any of the 906 labels, so `parse_diagnosis_name` is correct and
   is not to be retuned. That leaves open *why* the outputs miss. Comparing each fallback
   output against the 20 candidates that were in its own prompt separates "invented a
   condition" from "restated a candidate in shorter words".

Writes `artefacts/arm3_prompt_design_summary.csv` and
`artefacts/arm3_fallback_format_audit.csv`.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlh.config import ARTEFACTS_DIR, TRAIN_CSV  # noqa: E402

# A fallback output counts as an abridged rendering when this share of its words are drawn
# from a single candidate label. Set high deliberately: the claim in the report is that the
# model shortened a candidate, not that it merely overlapped one.
CONTAINMENT_THRESHOLD = 0.8


def normalise(text: str) -> str:
    """Fold to bare lowercase words, so `skin_lightening` and `Skin lightening` compare equal."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def prompt_candidates(prompt: str) -> list[str]:
    """Recover the 20 shortlist labels from the prompt the model was actually sent.

    The candidates are the comma-separated block after `Diagnoses:`. Reading them back out
    of the prompt rather than re-deriving them from the index means the audit compares each
    output against the exact strings that were in front of the model.
    """
    match = re.search(r"Diagnoses:\n(.*?)\n\n", str(prompt), re.S)
    if not match:
        return []
    return [candidate.strip() for candidate in match.group(1).split(",") if candidate.strip()]


def decompose(frame: pd.DataFrame) -> dict[str, float]:
    """Split one prompt-by-model cell into the three strata Arm 3 can land in.

    `fell_back` and `kept_rank_1` are inert by construction — both return Arm 1's top-1 — so
    every accuracy difference between conditions has to come from `moved_off_rank_1`.
    """
    fell_back = frame["fallback_fired"].astype(bool)
    kept = (~fell_back) & (frame["pred"] == frame["arm1_pred"])
    moved = (~fell_back) & (frame["pred"] != frame["arm1_pred"])

    overrides = frame[moved]
    rescued = ((overrides["pred"] == overrides["gold"]) & (overrides["arm1_pred"] != overrides["gold"])).sum()
    broke = ((overrides["pred"] != overrides["gold"]) & (overrides["arm1_pred"] == overrides["gold"])).sum()

    return {
        "n": len(frame),
        "accuracy": (frame["pred"] == frame["gold"]).mean(),
        "arm1_accuracy": (frame["arm1_pred"] == frame["gold"]).mean(),
        "kept_rank_1": int(kept.sum()),
        "fell_back": int(fell_back.sum()),
        "fallback_rate": fell_back.mean(),
        "moved_off_rank_1": int(moved.sum()),
        "override_rate": moved.mean(),
        "override_rescued": int(rescued),
        "override_broke": int(broke),
        "override_precision": rescued / len(overrides) if len(overrides) else float("nan"),
        "override_net": int(rescued) - int(broke),
    }


def audit_fallbacks(frame: pd.DataFrame, label_lookup: set[str]) -> dict[str, float]:
    """Characterise why the parser rejected each fallback output.

    `exact_label_match` re-checks CLAUDE.md's recorded finding under generous normalisation —
    it must stay 0, otherwise the fallbacks would be a parser defect rather than a finding.

    `abridged_candidate` counts outputs whose words are nearly all drawn from one candidate
    that was in the prompt: the model named the right kind of thing but did not copy the
    string. Not meaningful for chain-of-thought, whose `raw_output` is a whole reasoning
    trace, so the surrounding prose dilutes the overlap; reported anyway and flagged here.
    """
    fallbacks = frame[frame["fallback_fired"].astype(bool)]
    if not len(fallbacks):
        return {"n_fallback": 0, "exact_label_match": 0, "abridged_candidate": 0, "abridged_share": float("nan")}

    exact = 0
    abridged = 0
    for row in fallbacks.itertuples(index=False):
        output_words = normalise(row.raw_output).split()
        if not output_words:
            continue
        if normalise(row.raw_output) in label_lookup:
            exact += 1
        best = max(
            (
                sum(word in set(normalise(candidate).split()) for word in output_words) / len(output_words)
                for candidate in prompt_candidates(row.prompt)
            ),
            default=0.0,
        )
        if best >= CONTAINMENT_THRESHOLD:
            abridged += 1

    return {
        "n_fallback": len(fallbacks),
        "exact_label_match": exact,
        "abridged_candidate": abridged,
        "abridged_share": abridged / len(fallbacks),
    }


def main() -> None:
    predictions = pd.read_csv(ARTEFACTS_DIR / "arm3_val_predictions.csv")
    conditions = pd.read_csv(ARTEFACTS_DIR / "arm3_prompt_conditions.csv")
    labels = {normalise(label) for label in pd.read_csv(TRAIN_CSV)["disease"].unique()}
    print(f"{len(labels)} distinct labels; {len(predictions)} rows over "
          f"{predictions.groupby(['model_name', 'condition']).ngroups} prompt-by-model cells\n")

    wall_clock = conditions.set_index(["model_name", "condition"])["wall_clock_sec"]

    summary_rows, audit_rows = [], []
    for (model_name, condition), cell in predictions.groupby(["model_name", "condition"], sort=False):
        summary_rows.append(
            {
                "model_name": model_name,
                "condition": condition,
                **decompose(cell),
                "wall_clock_sec": wall_clock.get((model_name, condition), float("nan")),
            }
        )
        audit_rows.append({"model_name": model_name, "condition": condition, **audit_fallbacks(cell, labels)})

    summary = pd.DataFrame(summary_rows)
    audit = pd.DataFrame(audit_rows)

    print("=== prompt-design summary ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== fallback format audit ===")
    print(audit.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if audit["exact_label_match"].sum() != 0:
        raise ValueError(
            "a fallback output matches a real label — that would make the fallback rate a "
            "parser defect, contradicting the recorded finding. Investigate before reporting."
        )
    print("\nno fallback output in any cell matches one of the 906 labels: parser confirmed correct")

    summary.to_csv(ARTEFACTS_DIR / "arm3_prompt_design_summary.csv", index=False)
    audit.to_csv(ARTEFACTS_DIR / "arm3_fallback_format_audit.csv", index=False)
    print(f"wrote arm3_prompt_design_summary.csv and arm3_fallback_format_audit.csv to {ARTEFACTS_DIR}")


if __name__ == "__main__":
    main()
