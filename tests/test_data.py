from amlh.config import SEED
from amlh.data import load_test, load_train, make_validation_split, run_integrity_audit


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
