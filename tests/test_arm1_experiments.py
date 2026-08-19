from dataclasses import dataclass

import pandas as pd
import pytest

from amlh import arm1_experiments as ae

VEC_KWARGS = {"ngram_range": (1, 1), "sublinear_tf": False, "min_df": 1, "stop_words": None}


@dataclass(frozen=True)
class _FrozenHP:
    ngram_range: tuple = (1, 1)
    sublinear_tf: bool = False
    min_df: int = 1
    stop_words: object = None
    index_variant: str = "Q"
    index_scheme: str = "class_blob"
    k_neighbors: int = 1


@pytest.fixture
def tiny_split(split):
    """4 real classes, 1 row per class held out as val, the rest as fit."""
    diseases = split.fit.disease.value_counts().index[:4].tolist()
    subset = split.fit[split.fit.disease.isin(diseases)].reset_index(drop=True)
    val_small = subset.groupby("disease", group_keys=False)[subset.columns].apply(lambda g: g.tail(1))
    fit_small = subset.drop(val_small.index).reset_index(drop=True)
    val_small = val_small.reset_index(drop=True)
    return fit_small, val_small


def test_run_vectoriser_grid_smoke(tiny_split):
    fit_small, val_small = tiny_split
    df = ae.run_vectoriser_grid(
        fit_small,
        val_small,
        "Q",
        ngram_ranges=[(1, 1)],
        sublinear_tf_opts=[False],
        min_dfs=[1],
        stop_words_opts=[None],
        ks=[1, 3],
    )
    assert len(df) == 2
    expected_cols = {
        "ngram_range", "sublinear_tf", "min_df", "stop_words", "k",
        "accuracy", "acc_at_5", "macro_f1", "mrr",
    }
    assert expected_cols <= set(df.columns)


def test_select_within_one_se_prefers_simplest():
    grid = pd.DataFrame(
        [
            {"accuracy": 0.80, "k": 20, "ngram_range": (1, 2), "stop_words": "english", "min_df": 2},
            {"accuracy": 0.81, "k": 1, "ngram_range": (1, 1), "stop_words": None, "min_df": 1},
            {"accuracy": 0.60, "k": 1, "ngram_range": (1, 1), "stop_words": None, "min_df": 1},
        ]
    )
    chosen = ae.select_within_one_se(grid, n_val=200)
    assert chosen["k"] == 1
    assert chosen["ngram_range"] == (1, 1)
    assert chosen["stop_words"] is None


def test_run_preprocessing_ablation_smoke(tiny_split):
    fit_small, val_small = tiny_split
    df = ae.run_preprocessing_ablation(fit_small, val_small, VEC_KWARGS, k=1)
    assert list(df["preprocessing"]) == ["raw", "lemma_stop", "lemma_only"]


def test_run_index_variant_comparison_smoke(tiny_split):
    fit_small, val_small = tiny_split
    df = ae.run_index_variant_comparison(fit_small, val_small, ["Q", "QL"], VEC_KWARGS, k=1)
    assert list(df["variant"]) == ["Q", "QL"]


def test_run_indexing_scheme_comparison_smoke(tiny_split):
    fit_small, val_small = tiny_split
    df = ae.run_indexing_scheme_comparison(fit_small, val_small, "QL", VEC_KWARGS, k=1)
    assert set(df["scheme"]) == {"class_blob", "additive_per_row"}

    blob_rows = df.loc[df.scheme == "class_blob", "n_index_rows"].item()
    additive_rows = df.loc[df.scheme == "additive_per_row", "n_index_rows"].item()
    n_classes = fit_small.disease.nunique()
    assert blob_rows == n_classes
    assert additive_rows == len(fit_small) + n_classes  # "Q" per-row + "L" one row/class


def test_run_variant_scheme_grid_smoke(tiny_split):
    fit_small, val_small = tiny_split
    df = ae.run_variant_scheme_grid(fit_small, val_small, ["Q", "QL"], VEC_KWARGS, k=1)
    assert len(df) == 4  # 2 schemes x 2 variants
    assert set(df["scheme"]) == {"class_blob", "additive_per_row"}
    assert set(df["variant"]) == {"Q", "QL"}
    expected_cols = {"scheme", "variant", "accuracy", "acc_at_5", "macro_f1", "mrr"}
    assert expected_cols <= set(df.columns)


def test_run_split_robustness_smoke(train):
    configs = [
        {"ngram_range": (1, 1), "sublinear_tf": False, "min_df": 1, "stop_words": None, "k": 1},
        {"ngram_range": (1, 1), "sublinear_tf": False, "min_df": 1, "stop_words": None, "k": 3},
    ]
    df = ae.run_split_robustness(train, configs, variant="Q", seeds=(42, 43))
    assert len(df) == len(configs) * 2
    assert set(df["seed"]) == {42, 43}
    assert set(df["config_rank"]) == {1, 2}


def test_run_supervised_baselines_smoke(tiny_split):
    fit_small, val_small = tiny_split
    df = ae.run_supervised_baselines(fit_small, val_small, VEC_KWARGS)
    assert list(df["model"]) == ["svc", "logreg", "random_forest"]


def test_frozen_ranking_depth_none_matches_grid(tiny_split):
    fit_small, val_small = tiny_split
    hp = _FrozenHP()
    ranked, top_sim = ae.frozen_ranking(fit_small, val_small, hp)
    grid = ae.run_vectoriser_grid(
        fit_small, val_small, hp.index_variant, [hp.ngram_range], [hp.sublinear_tf], [hp.min_df],
        [hp.stop_words], [hp.k_neighbors],
    )
    assert grid.iloc[0]["accuracy"] == sum(
        r[0] == g for r, g in zip(ranked, val_small["disease"])
    ) / len(val_small)
    assert len(top_sim) == len(val_small)


def test_build_val_predictions_schema_and_top1_matches_frozen(tiny_split):
    fit_small, val_small = tiny_split
    hp = _FrozenHP()
    depth = 4
    ranked_frozen, _ = ae.frozen_ranking(fit_small, val_small, hp)
    preds = ae.build_val_predictions(fit_small, val_small, hp, depth=depth)

    assert len(preds) == len(val_small)
    expected_cols = {"question", "gold", "pred", "top_sim", *(f"top_{i}" for i in range(1, depth + 1))}
    assert expected_cols <= set(preds.columns)
    assert preds["pred"].tolist() == [r[0] for r in ranked_frozen]
    assert preds["question"].tolist() == val_small["question"].tolist()
    assert preds["gold"].tolist() == val_small["disease"].tolist()


def test_shortlist_ceiling_monotonic_and_matches_accuracy_at_k(tiny_split):
    fit_small, val_small = tiny_split
    hp = _FrozenHP()
    preds = ae.build_val_predictions(fit_small, val_small, hp, depth=4)
    ceiling = ae.shortlist_ceiling(preds, [1, 2, 4])

    assert list(ceiling["k"]) == [1, 2, 4]
    assert ceiling["accuracy"].is_monotonic_increasing
    top1_acc = (preds["pred"] == val_small["disease"].reset_index(drop=True)).mean()
    assert ceiling.loc[ceiling["k"] == 1, "accuracy"].item() == pytest.approx(top1_acc)
