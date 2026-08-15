import pandas as pd
import pytest

from amlh import arm1_experiments as ae

VEC_KWARGS = {"ngram_range": (1, 1), "sublinear_tf": False, "min_df": 1, "stop_words": None}


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
