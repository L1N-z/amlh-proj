from unittest.mock import patch

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from amlh.arm1_tfidf import knn_rank
from amlh.features import build_index, build_vectoriser, lemmatise, load_class_doc


def test_build_vectoriser_passes_through_kwargs():
    vec = build_vectoriser(ngram_range=(1, 2), sublinear_tf=True)
    assert isinstance(vec, TfidfVectorizer)
    assert vec.ngram_range == (1, 2)
    assert vec.sublinear_tf is True


def test_knn_rank_fits_only_on_index_not_query(split):
    fit_small = split.fit.head(20)
    texts, labels = build_index(fit_small, "Q")
    queries = split.val.question.head(5).tolist()

    with (
        patch.object(
            TfidfVectorizer, "fit_transform", wraps=TfidfVectorizer.fit_transform, autospec=True
        ) as spy_fit,
        patch.object(
            TfidfVectorizer, "transform", wraps=TfidfVectorizer.transform, autospec=True
        ) as spy_transform,
    ):
        knn_rank(texts, labels, queries, k=5, vec_kwargs={})

    fit_call_arg = spy_fit.call_args[0][1]
    transform_call_arg = spy_transform.call_args[0][1]
    assert list(fit_call_arg) == texts
    assert list(transform_call_arg) == queries


def test_build_index_row_count_equals_n_classes(split):
    n_classes = split.fit.disease.nunique()
    for variant in ("Q", "QL", "QLA", "QLAD"):
        texts, labels = build_index(split.fit, variant)
        assert len(texts) == len(labels) == n_classes


def test_build_index_variant_order_independent(split):
    a, la = build_index(split.fit, "QL")
    b, lb = build_index(split.fit, "LQ")
    assert a == b
    assert la == lb


def test_build_index_rejects_test_frame_when_variant_needs_answer(test):
    with pytest.raises(KeyError):
        build_index(test, "QLA")


def test_build_index_invalid_variant_raises(split):
    with pytest.raises(ValueError):
        build_index(split.fit, "QZ")


def test_build_index_labels_subset_of_disease_universe(train, split):
    _, labels = build_index(split.fit, "QLAD")
    assert set(labels) <= set(train.disease.unique())


@pytest.mark.parametrize("disease", ["Bronchitis", "Multiple_sclerosis", "abscess"])
def test_load_class_doc_resolves_known_labels(disease):
    text = load_class_doc(disease)
    assert text != ""


def test_load_class_doc_strips_boilerplate():
    text = load_class_doc("abscess")
    for pattern in (
        "Skip to main content",
        "Page last reviewed",
        "Next review due",
        "Credit:",
        "alamy.com",
        "- nhs",
    ):
        assert pattern.lower() not in text.lower()
    assert "abscess" in text.lower()
    assert "pus" in text.lower()


def test_load_class_doc_missing_returns_empty():
    assert load_class_doc("this_disease_does_not_exist") == ""


def _spacy_ready() -> bool:
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _spacy_ready(), reason="spacy/en_core_web_sm not installed yet")
def test_lemmatise_deterministic_and_length_preserving():
    texts = ["Running quickly to the shops", "The cats are sleeping", ""]
    a = lemmatise(texts)
    b = lemmatise(texts)
    assert a == b
    assert len(a) == len(texts)
