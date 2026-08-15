"""Validation-only diagnostic: does the four-way accuracy tie across the
class_blob index variants (Q, QL, QLA, QLAD) at std-val hold across different
seeds of make_validation_split, or was seed=42 a coincidence?

No test-set access. No config.py changes. Seed 44 is reported as the headline
("selection basis") result; the other seeds are robustness evidence only —
config.py's frozen SEED=42 split (and split_fit.csv/split_val.csv in
artefacts/) is untouched by this script.
"""

import pandas as pd

from amlh import arm1_tfidf, features
from amlh.config import HYPERPARAMETERS, set_seed
from amlh.data import load_train, make_validation_split
from amlh.evaluate import pairwise_top1_disagreement

VARIANTS = ["Q", "QL", "QLA", "QLAD"]
SEEDS = [42, 43, 44, 45, 46]
HEADLINE_SEED = 44
NEAR_EQUALITY_THRESHOLD = (0.82 * (1 - 0.82) / 200) ** 0.5  # 1 SE at std-val's n=200, acc~0.82


def _vec_kwargs() -> dict:
    return {
        "ngram_range": HYPERPARAMETERS.ngram_range,
        "sublinear_tf": HYPERPARAMETERS.sublinear_tf,
        "min_df": HYPERPARAMETERS.min_df,
        "stop_words": HYPERPARAMETERS.stop_words,
    }


def _class_blob_predictions(fit_df, val_questions, vec_kwargs, k) -> dict[str, list]:
    top1_by_variant = {}
    for variant in VARIANTS:
        index_texts, index_labels = features.build_index(fit_df, variant)
        ranked, _ = arm1_tfidf.knn_rank(index_texts, index_labels, val_questions, k, vec_kwargs)
        top1_by_variant[variant] = [r[0] if r else None for r in ranked]
    return top1_by_variant


def main() -> None:
    set_seed()
    train = load_train()
    vec_kwargs = _vec_kwargs()
    k = HYPERPARAMETERS.k_neighbors

    print(f"frozen Arm 1 vectoriser config: {vec_kwargs} | k={k}")
    print(f"seeds tested: {SEEDS} | headline/selection-basis seed: {HEADLINE_SEED}")
    print()

    rows = []
    top1_by_seed = {}
    for seed in SEEDS:
        std_split = make_validation_split(train, seed=seed)
        gold = std_split.val["disease"].tolist()
        top1_by_variant = _class_blob_predictions(
            std_split.fit, std_split.val["question"].tolist(), vec_kwargs, k
        )
        top1_by_seed[seed] = top1_by_variant
        for variant, top1 in top1_by_variant.items():
            accuracy = sum(p == g for p, g in zip(top1, gold)) / len(gold)
            rows.append({"seed": seed, "variant": variant, "accuracy": accuracy})

    acc_table = pd.DataFrame(rows)
    pivot = acc_table.pivot(index="seed", columns="variant", values="accuracy")[VARIANTS]
    pivot["spread"] = pivot.max(axis=1) - pivot.min(axis=1)
    print("=== std-val accuracy by seed x class_blob variant ===")
    print(pivot.to_string())
    print()

    max_spread = pivot["spread"].max()
    print(
        f"max per-seed spread across variants: {max_spread:.4f} "
        f"(1 SE reference at n=200, acc~0.82: {NEAR_EQUALITY_THRESHOLD:.4f})"
    )
    persists = bool(max_spread <= NEAR_EQUALITY_THRESHOLD)
    print(f"near-equality persists across all {len(SEEDS)} seeds: {persists}")
    print()

    print(f"=== changed-prediction matrix at headline seed={HEADLINE_SEED} (class_blob) ===")
    headline_matrix = pairwise_top1_disagreement(top1_by_seed[HEADLINE_SEED])
    print(headline_matrix.to_string())
    print()

    headline_mean_acc = pivot.loc[HEADLINE_SEED].drop("spread").mean()

    if persists:
        label = "saturation"
        prose = (
            f'"saturation" -- across all {len(SEEDS)} seeds tested (headline seed={HEADLINE_SEED}), '
            f"the four class_blob index variants remain within {max_spread:.3f} of each other on "
            f"std-val accuracy (<= 1 SE reference, {NEAR_EQUALITY_THRESHOLD:.3f}), despite the "
            f"changed-prediction matrix above showing real, non-trivial per-item disagreement "
            f"between variants. std-val accuracy is saturated at ~{headline_mean_acc:.3f} regardless "
            f"of index variant; it is not measuring which variant is better."
        )
    else:
        label = "fails to discriminate"
        prose = (
            f'"fails to discriminate" -- near-equality does NOT persist across all {len(SEEDS)} '
            f"seeds (max spread {max_spread:.3f} exceeds the 1 SE reference, "
            f"{NEAR_EQUALITY_THRESHOLD:.3f}); seed=42's exact tie looks like a property of that "
            f"particular split, not a stable one. Lead on the changed-prediction matrix at the "
            f"headline seed (above) instead: std-val variants disagree on a real fraction of items "
            f"per seed, it simply lacks the power at n=200 to turn that disagreement into a "
            f"resolvable accuracy difference."
        )

    print(f"write-up label: {label}")
    print(prose)
    print()
    print("config.py's frozen SEED=42 and Arm 1 hyperparameters were NOT changed by this script.")


if __name__ == "__main__":
    main()
