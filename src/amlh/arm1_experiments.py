"""Sweep and ablation orchestration for Arm 1 (TF-IDF / k-NN).

Every function here wraps already-tested primitives in `features`,
`arm1_tfidf`, and `evaluate` — no new indexing or scoring logic lives here,
just the loops that sweep them over hyperparameter/variant/scheme grids.
"""

import math
from itertools import product

import pandas as pd

from amlh import arm1_tfidf, data, features
from amlh.evaluate import accuracy_at_k, score_ranked


def _score_row(ranked: list[list[str]], gold: list[str], extra: dict) -> dict:
    return {**extra, **score_ranked(ranked, gold)}


def run_vectoriser_grid(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    variant: str,
    ngram_ranges: list[tuple[int, int]],
    sublinear_tf_opts: list[bool],
    min_dfs: list[int],
    stop_words_opts: list,
    ks: list[int],
) -> pd.DataFrame:
    """Sweep vectoriser hyperparameters x k on a class-blob `variant` index.
    The index is built once per vectoriser combo (it doesn't depend on k)."""
    gold = val_df["disease"].tolist()
    queries = val_df["question"].tolist()
    rows = []
    for ngram_range, sublinear_tf, min_df, stop_words in product(
        ngram_ranges, sublinear_tf_opts, min_dfs, stop_words_opts
    ):
        vec_kwargs = {
            "ngram_range": ngram_range,
            "sublinear_tf": sublinear_tf,
            "min_df": min_df,
            "stop_words": stop_words,
        }
        index_texts, index_labels = features.build_index(fit_df, variant)
        for k in ks:
            ranked, _ = arm1_tfidf.knn_rank(index_texts, index_labels, queries, k, vec_kwargs)
            rows.append(_score_row(ranked, gold, {**vec_kwargs, "k": k}))
    return pd.DataFrame(rows)


def select_within_one_se(grid_df: pd.DataFrame, n_val: int = 200) -> pd.Series:
    """Among configs within 1 SE of the top validation accuracy, return the
    simplest: lowest k, ngram_range=(1,1) over (1,2), stop_words=None over
    "english", min_df=1 as a final tiebreak. Guards against picking the raw
    argmax of a noisy 200-item hold-out."""
    best_acc = grid_df["accuracy"].max()
    se = math.sqrt(best_acc * (1 - best_acc) / n_val)
    within = grid_df[grid_df["accuracy"] >= best_acc - se].copy()
    within["_ngram_penalty"] = within["ngram_range"].apply(lambda ng: 0 if tuple(ng) == (1, 1) else 1)
    within["_stopword_penalty"] = within["stop_words"].apply(lambda s: 0 if s is None else 1)
    within = within.sort_values(
        by=["k", "_ngram_penalty", "_stopword_penalty", "min_df"],
        ascending=[True, True, True, True],
    )
    return within.iloc[0].drop(["_ngram_penalty", "_stopword_penalty"])


def run_preprocessing_ablation(fit_df: pd.DataFrame, val_df: pd.DataFrame, vec_kwargs: dict, k: int) -> pd.DataFrame:
    """raw / lemma_stop / lemma_only at variant "Q" — matches the grid this
    ablation is run on top of. Lemmatised text is computed once per
    drop_stop setting and reused, not recomputed per row."""
    gold = val_df["disease"].tolist()
    raw_queries = val_df["question"].tolist()

    lemma_stop_fit_q = features.lemmatise(fit_df["question"].tolist(), drop_stop=True)
    lemma_only_fit_q = features.lemmatise(fit_df["question"].tolist(), drop_stop=False)
    lemma_stop_val_q = features.lemmatise(raw_queries, drop_stop=True)
    lemma_only_val_q = features.lemmatise(raw_queries, drop_stop=False)

    rows = []

    index_texts, index_labels = features.build_index(fit_df, "Q")
    ranked, _ = arm1_tfidf.knn_rank(index_texts, index_labels, raw_queries, k, vec_kwargs)
    rows.append(_score_row(ranked, gold, {"preprocessing": "raw"}))

    for name, fit_q, val_q in (
        ("lemma_stop", lemma_stop_fit_q, lemma_stop_val_q),
        ("lemma_only", lemma_only_fit_q, lemma_only_val_q),
    ):
        fit_lemma = fit_df.copy()
        fit_lemma["question"] = fit_q
        index_texts, index_labels = features.build_index(fit_lemma, "Q")
        ranked, _ = arm1_tfidf.knn_rank(index_texts, index_labels, val_q, k, vec_kwargs)
        rows.append(_score_row(ranked, gold, {"preprocessing": name}))

    return pd.DataFrame(rows)


def run_index_variant_comparison(
    fit_df: pd.DataFrame, val_df: pd.DataFrame, variants: list[str], vec_kwargs: dict, k: int
) -> pd.DataFrame:
    gold = val_df["disease"].tolist()
    queries = val_df["question"].tolist()
    rows = []
    for variant in variants:
        index_texts, index_labels = features.build_index(fit_df, variant)
        ranked, _ = arm1_tfidf.knn_rank(index_texts, index_labels, queries, k, vec_kwargs)
        rows.append(_score_row(ranked, gold, {"variant": variant}))
    return pd.DataFrame(rows)


def run_indexing_scheme_comparison(
    fit_df: pd.DataFrame, val_df: pd.DataFrame, variant: str, vec_kwargs: dict, k: int
) -> pd.DataFrame:
    gold = val_df["disease"].tolist()
    queries = val_df["question"].tolist()
    rows = []
    for scheme, builder in (
        ("class_blob", features.build_index),
        ("additive_per_row", features.build_index_additive),
    ):
        index_texts, index_labels = builder(fit_df, variant)
        ranked, _ = arm1_tfidf.knn_rank(index_texts, index_labels, queries, k, vec_kwargs)
        rows.append(_score_row(ranked, gold, {"scheme": scheme, "n_index_rows": len(index_texts)}))
    return pd.DataFrame(rows)


def run_variant_scheme_grid(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    variants: list[str],
    vec_kwargs: dict,
    k: int,
    schemes: tuple[str, ...] = ("class_blob", "additive_per_row"),
) -> pd.DataFrame:
    """Score every (scheme, variant) combination on a single (fit_df, val_df)
    pair. Agnostic to what val_df is — standard val, hard val, or test; the
    caller decides that, this function only runs the grid."""
    builders = {"class_blob": features.build_index, "additive_per_row": features.build_index_additive}
    gold = val_df["disease"].tolist()
    queries = val_df["question"].tolist()
    rows = []
    for scheme in schemes:
        builder = builders[scheme]
        for variant in variants:
            index_texts, index_labels = builder(fit_df, variant)
            ranked, _ = arm1_tfidf.knn_rank(index_texts, index_labels, queries, k, vec_kwargs)
            rows.append(_score_row(ranked, gold, {"scheme": scheme, "variant": variant}))
    return pd.DataFrame(rows)


def run_split_robustness(
    train_df: pd.DataFrame,
    configs: list[dict],
    variant: str = "Q",
    seeds: tuple[int, ...] = (42, 43, 44),
) -> pd.DataFrame:
    """Rerun `configs` (top-N rows from a vectoriser grid, as dicts with
    ngram_range/sublinear_tf/min_df/stop_words/k) under an unstratified
    `make_random_split` for each seed. One row per (config_rank, seed).
    Absolute accuracy is not comparable to the stratified-split grid — only
    whether the ranking across `configs` is preserved is meaningful here."""
    rows = []
    for seed in seeds:
        split = data.make_random_split(train_df, seed=seed)
        gold = split.val["disease"].tolist()
        queries = split.val["question"].tolist()
        index_texts, index_labels = features.build_index(split.fit, variant)
        for rank, cfg in enumerate(configs, start=1):
            vec_kwargs = {
                "ngram_range": cfg["ngram_range"],
                "sublinear_tf": cfg["sublinear_tf"],
                "min_df": cfg["min_df"],
                "stop_words": cfg["stop_words"],
            }
            ranked, _ = arm1_tfidf.knn_rank(index_texts, index_labels, queries, cfg["k"], vec_kwargs)
            rows.append(
                _score_row(ranked, gold, {"config_rank": rank, "seed": seed, **vec_kwargs, "k": cfg["k"]})
            )
    return pd.DataFrame(rows)


def frozen_ranking(
    fit_df: pd.DataFrame, val_df: pd.DataFrame, hp, depth: int | None = None
) -> tuple[list[list[str]], list[float]]:
    """Run `arm1_tfidf.knn_rank` at the frozen Arm 1 hyperparameters in `hp`
    (normally `config.HYPERPARAMETERS`) over `val_df`. `depth=None` reproduces
    exactly the ranking every grid in this module already reports; passing a
    `depth` extends it for cross-arm comparability without touching the vote
    that produces rank 1 — see `arm1_tfidf.knn_rank`. The single place that
    reads `hp`'s vectoriser/index/k fields, so the frozen-path comparison and
    the depth-extended path can never read them differently."""
    vec_kwargs = {
        "ngram_range": hp.ngram_range,
        "sublinear_tf": hp.sublinear_tf,
        "min_df": hp.min_df,
        "stop_words": hp.stop_words,
    }
    build_fn = features.build_index if hp.index_scheme == "class_blob" else features.build_index_additive
    index_texts, index_labels = build_fn(fit_df, hp.index_variant)
    return arm1_tfidf.knn_rank(
        index_texts, index_labels, val_df["question"].tolist(), hp.k_neighbors, vec_kwargs, depth=depth
    )


def build_val_predictions(fit_df: pd.DataFrame, val_df: pd.DataFrame, hp, depth: int) -> pd.DataFrame:
    """Per-item Arm 1 validation predictions at the frozen hyperparameters in
    `hp`, ranked `depth` labels deep, aligned to `val_df`'s row order. No
    aggregate Arm 1 grid records per-item output, so McNemar against Arm 2 and
    the error analysis (most-confused pairs, family-internal errors, worked
    examples) have nothing to read without this. Columns short of `depth`
    ranked labels are left empty rather than padded with a placebo label."""
    ranked, top_sim = frozen_ranking(fit_df, val_df, hp, depth=depth)
    rows = []
    for question, gold, r, sim in zip(val_df["question"], val_df["disease"], ranked, top_sim):
        row = {"question": question, "gold": gold, "pred": r[0]}
        for i in range(depth):
            row[f"top_{i + 1}"] = r[i] if i < len(r) else None
        row["top_sim"] = sim
        rows.append(row)
    return pd.DataFrame(rows)


def shortlist_ceiling(predictions_df: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    """acc@k, for each k in `ks`, over a `build_val_predictions` frame's
    top_1..top_N columns — the bound Arm 3's shortlist-then-select design is
    interpreted against. Reuses `evaluate.accuracy_at_k` rather than scoring
    top-k membership a second way."""
    top_cols = [f"top_{i}" for i in range(1, max(ks) + 1)]
    ranked = [[lab for lab in row if isinstance(lab, str)] for row in predictions_df[top_cols].values.tolist()]
    gold = predictions_df["gold"].tolist()
    return pd.DataFrame({"k": ks, "accuracy": [accuracy_at_k(ranked, gold, k) for k in ks]})


def run_supervised_baselines(fit_df: pd.DataFrame, val_df: pd.DataFrame, vec_kwargs: dict) -> pd.DataFrame:
    """Per-row (not blob) TF-IDF + LinearSVC / LogisticRegression /
    RandomForestClassifier, trained on individual fit questions."""
    gold = val_df["disease"].tolist()
    queries = val_df["question"].tolist()
    train_texts = fit_df["question"].tolist()
    train_labels = fit_df["disease"].tolist()
    rows = []
    for model in ("svc", "logreg", "random_forest"):
        ranked, _ = arm1_tfidf.linear_rank(train_texts, train_labels, queries, vec_kwargs, model)
        rows.append(_score_row(ranked, gold, {"model": model}))
    return pd.DataFrame(rows)
