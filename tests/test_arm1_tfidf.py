import pytest

from amlh import features
from amlh.arm1_tfidf import knn_rank, linear_rank

VEC_KWARGS = {"ngram_range": (1, 1), "sublinear_tf": False, "min_df": 1, "stop_words": None}

_INDEX_TEXTS = [
    "diabetes causes fatigue thirst and weight loss over months",
    "asthma causes wheezing and shortness of breath during exertion",
    "the flu causes fever cough sore throat and muscle aches quickly",
    "migraine causes throbbing headache aura and nausea sometimes",
    "eczema causes itchy dry skin rash on the hands",
]
_INDEX_LABELS = ["diabetes", "asthma", "flu", "migraine", "eczema"]
_QUERIES = ["why do I have fatigue and weight loss", "why do I have fever cough and sore throat"]
# Cosine distances among the top 3 neighbours of each query are pairwise distinct (checked by
# hand) for k in {1, 2, 3} below. sklearn's brute NearestNeighbors uses np.argpartition
# internally, whose tie-break order for exactly-equal distances is not guaranteed stable across
# different n_neighbors values — an index with a genuine tie would make the "head preserved"
# assertion flaky for reasons unrelated to knn_rank's own logic.


@pytest.mark.parametrize("k", [1, 2, 3])
def test_knn_rank_depth_none_matches_no_depth_arg(k):
    with_default = knn_rank(_INDEX_TEXTS, _INDEX_LABELS, _QUERIES, k, VEC_KWARGS)
    explicit_none = knn_rank(_INDEX_TEXTS, _INDEX_LABELS, _QUERIES, k, VEC_KWARGS, depth=None)
    assert with_default == explicit_none


@pytest.mark.parametrize("k", [1, 2, 3])
def test_knn_rank_depth_preserves_head(k):
    """The depth-extended ranking's head must equal the no-depth ranking
    exactly — not just as a set — for any k, not only k=1."""
    ranked_base, top_sim_base = knn_rank(_INDEX_TEXTS, _INDEX_LABELS, _QUERIES, k, VEC_KWARGS)
    ranked_deep, top_sim_deep = knn_rank(
        _INDEX_TEXTS, _INDEX_LABELS, _QUERIES, k, VEC_KWARGS, depth=len(_INDEX_TEXTS)
    )

    assert top_sim_base == top_sim_deep
    for base, deep in zip(ranked_base, ranked_deep):
        assert deep[: len(base)] == base
        assert len(deep) <= len(_INDEX_LABELS)
        assert len(set(deep)) == len(deep)  # no duplicate labels


def test_knn_rank_depth_smaller_than_k_is_a_no_op():
    ranked_base, _ = knn_rank(_INDEX_TEXTS, _INDEX_LABELS, _QUERIES, 3, VEC_KWARGS)
    ranked_shallow, _ = knn_rank(_INDEX_TEXTS, _INDEX_LABELS, _QUERIES, 3, VEC_KWARGS, depth=1)
    assert ranked_base == ranked_shallow


def test_knn_rank_depth_clips_to_index_size():
    ranked, _ = knn_rank(_INDEX_TEXTS, _INDEX_LABELS, _QUERIES, 1, VEC_KWARGS, depth=1000)
    for r in ranked:
        assert len(r) <= len(_INDEX_LABELS)
        assert set(r) <= set(_INDEX_LABELS)


def test_knn_rank_depth_top1_identical_on_real_class_blob_index(split):
    """Mirrors the actual notebook usage: a real class-blob index at the
    frozen k_neighbors=1, comparing depth=None against depth=20 top-1."""
    index_texts, index_labels = features.build_index(split.fit, "Q")
    queries = split.val["question"].tolist()

    ranked_frozen, top_sim_frozen = knn_rank(index_texts, index_labels, queries, 1, VEC_KWARGS)
    ranked_deep, top_sim_deep = knn_rank(index_texts, index_labels, queries, 1, VEC_KWARGS, depth=20)

    top1_frozen = [r[0] for r in ranked_frozen]
    top1_deep = [r[0] for r in ranked_deep]
    assert top1_frozen == top1_deep
    assert top_sim_frozen == top_sim_deep


@pytest.mark.parametrize("model", ["svc", "logreg", "random_forest"])
def test_linear_rank_ranks_all_classes_deterministically(split, model):
    fit_small = split.fit.head(60)
    train_texts = fit_small.question.tolist()
    train_labels = fit_small.disease.tolist()
    queries = split.val.question.head(5).tolist()
    n_classes = len(set(train_labels))

    ranked_a, scores_a = linear_rank(train_texts, train_labels, queries, {}, model, seed=42)
    ranked_b, scores_b = linear_rank(train_texts, train_labels, queries, {}, model, seed=42)

    assert len(ranked_a) == len(queries)
    for r in ranked_a:
        assert len(r) == n_classes
        assert set(r) == set(train_labels)
    assert ranked_a == ranked_b
    assert scores_a == pytest.approx(scores_b)


def test_linear_rank_invalid_model_raises(split):
    fit_small = split.fit.head(10)
    with pytest.raises(ValueError):
        linear_rank(
            fit_small.question.tolist(),
            fit_small.disease.tolist(),
            split.val.question.head(2).tolist(),
            {},
            "not_a_model",
        )
