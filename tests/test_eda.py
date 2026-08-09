import numpy as np
import pandas as pd

from amlh.data import run_integrity_audit
from amlh.eda import (
    ambiguous_examples,
    label_family,
    near_duplication_similarities,
    novelty_calibrated_eval,
    sibling_homogeneity,
    wrong_class_closer_fraction,
)


def test_label_family_extracts_prefix():
    diseases = pd.Series(["baby_colic", "pregnancy_back_pain", "abscess"])
    fam = label_family(diseases)
    assert fam.tolist() == ["baby", "pregnancy", "abscess"]


def test_ambiguous_examples_only_multi_disease_questions(train):
    examples = ambiguous_examples(train, n=4)
    assert len(examples) <= 4
    for ex in examples:
        assert len(ex["diseases"]) > 1
        assert set(train.loc[train.question == ex["question"], "disease"]) == set(ex["diseases"])


def test_ambiguous_examples_respects_n(train):
    examples = ambiguous_examples(train, n=2)
    assert len(examples) == 2


def test_near_duplication_similarities_matches_audit_fractions(train, test):
    sims = near_duplication_similarities(train, test)
    assert sims.shape == (len(test),)
    assert ((sims >= 0) & (sims <= 1)).all()

    audit = run_integrity_audit(train, test)
    for threshold, key in ((0.95, "frac_cos_ge_0.95"), (0.7, "frac_cos_ge_0.7")):
        assert round(float((sims >= threshold).mean()), 4) == audit["near_duplication"][key]


def test_sibling_homogeneity_shapes_and_ranges(train, test):
    result = sibling_homogeneity(train, test)
    assert len(result.own) == len(test)
    assert len(result.other) == len(test)
    assert len(result.sib) > 0
    for values in (result.sib, result.own, result.other):
        assert all(-1.0 <= v <= 1.0 for v in values)
    assert len(result.summary) == 3
    assert set(result.summary.columns) == {"comparison", "mean_cosine"}


def test_wrong_class_closer_fraction():
    own = [0.5, 0.5, 0.3]
    other = [0.6, 0.4, 0.9]
    assert wrong_class_closer_fraction(own, other) == 2 / 3


def test_novelty_calibrated_eval_ranges(split):
    fit_small = split.fit.head(500)
    val_small = split.val.head(30)
    accuracy, novelty = novelty_calibrated_eval(fit_small, val_small, prune_threshold=1.01, k=20)
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= novelty <= 1.0


def test_novelty_calibrated_eval_pruning_does_not_increase_novelty(split):
    fit_small = split.fit.head(500)
    val_small = split.val.head(30)
    _, novelty_unpruned = novelty_calibrated_eval(fit_small, val_small, prune_threshold=1.01, k=20)
    _, novelty_pruned = novelty_calibrated_eval(fit_small, val_small, prune_threshold=0.4, k=20)
    assert novelty_pruned <= novelty_unpruned
