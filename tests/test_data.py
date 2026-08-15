import re

from amlh.config import SEED
from amlh.data import (
    load_test,
    load_train,
    make_hard_validation_split,
    make_validation_split,
    run_integrity_audit,
)


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\w+", text) if len(w) > 2}


def test_validation_classes_subset_of_fit(split):
    assert set(split.val.disease) <= set(split.fit.disease)


def test_every_class_retains_two_in_fit(split, train):
    counts = split.fit.disease.value_counts().reindex(train.disease.unique())
    assert counts.min() >= 2


def test_split_deterministic_under_seed(train):
    a = make_validation_split(train, seed=SEED)
    b = make_validation_split(train, seed=SEED)
    assert list(a.val.index) == list(b.val.index)
    assert a.val.equals(b.val)


def test_no_test_answer_leakage(train):
    test = load_test()
    assert "answer" not in test.columns
    run_integrity_audit(train, test.drop(columns=[c for c in test.columns if c == "answer"]))


def test_make_hard_validation_split_val_is_label_word_free(train):
    split = make_hard_validation_split(train, n=50)
    for row in split.val.itertuples():
        label_words = _content_words(row.disease.replace("_", " "))
        question_words = _content_words(row.question)
        assert question_words.isdisjoint(label_words), (row.question, row.disease)


def test_make_hard_validation_split_fit_val_disjoint(train):
    split = make_hard_validation_split(train, n=50)
    assert set(split.val.index) & set(split.fit.index) == set()
    assert set(split.val.disease) <= set(split.fit.disease)


def test_make_hard_validation_split_reproducible(train):
    a = make_hard_validation_split(train, seed=SEED, n=50)
    b = make_hard_validation_split(train, seed=SEED, n=50)
    assert list(a.val.index) == list(b.val.index)
    assert a.val.equals(b.val)
