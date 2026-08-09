import pytest

from amlh.arm1_tfidf import linear_rank


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
