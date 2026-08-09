"""Diagnostics for the validation-test gap and label-space structure.

``sibling_homogeneity`` and ``novelty_calibrated_eval`` read ``test.disease`` under
CLAUDE.md's declared 01_eda exception: they characterise the validation-test
relationship (sibling phrasing homogeneity, the novelty-calibration target) and
select no model, hyperparameter, preprocessing or index-variant choice.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def label_family(diseases: pd.Series) -> pd.Series:
    """Shared-prefix family per disease label (text before the first `_`)."""
    return diseases.str.split("_").str[0]


def ambiguous_examples(train: pd.DataFrame, n: int = 4) -> list[dict]:
    """Up to `n` example question strings that map to more than one disease."""
    dup = train[train.duplicated("question", keep=False)]
    grouped = dup.groupby("question").disease.unique()
    examples = []
    for question, diseases in grouped.items():
        if len(diseases) <= 1:
            continue
        examples.append({"question": question, "diseases": sorted(diseases)})
        if len(examples) >= n:
            break
    return examples


def near_duplication_similarities(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Per-test-question max cosine similarity to any train question.

    Same TF-IDF(1,2) recipe as ``data.run_integrity_audit``'s near-duplication
    check, exposed as an array (rather than summary fractions) for plotting.
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(train.question)
    sim = cosine_similarity(vec.transform(test.question), vec.transform(train.question))
    return sim.max(axis=1)


@dataclass
class SiblingHomogeneity:
    sib: list[float]  # train question -> nearest same-class sibling
    own: list[float]  # test question -> nearest own-class train question
    other: list[float]  # test question -> nearest other-class train question
    summary: pd.DataFrame


def sibling_homogeneity(train: pd.DataFrame, test: pd.DataFrame) -> SiblingHomogeneity:
    """Measure phrasing homogeneity within a class vs. between train and test.

    Declared diagnostic use of `test.disease` (CLAUDE.md hard rule #2 exception).
    """
    vec = TfidfVectorizer(sublinear_tf=True).fit(train.question)
    Xtr = vec.transform(train.question)
    Xte = vec.transform(test.question)

    sib: list[float] = []
    for _, idx in train.groupby("disease").groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        S = cosine_similarity(Xtr[idx])
        np.fill_diagonal(S, -1)
        sib.extend(S.max(axis=1).tolist())

    by_disease = train.groupby("disease").groups
    own = [
        float(cosine_similarity(Xte[i], Xtr[np.asarray(by_disease[d])]).max())
        for i, d in enumerate(test.disease)
    ]

    S_all = cosine_similarity(Xte, Xtr)
    train_disease = train.disease.to_numpy()
    other = [
        float(S_all[i][train_disease != d].max()) for i, d in enumerate(test.disease)
    ]

    summary = pd.DataFrame(
        [
            {
                "comparison": "train question -> nearest same-class sibling",
                "mean_cosine": round(float(np.mean(sib)), 3),
            },
            {
                "comparison": "test question -> nearest own-class train question",
                "mean_cosine": round(float(np.mean(own)), 3),
            },
            {
                "comparison": "test question -> nearest other-class train question",
                "mean_cosine": round(float(np.mean(other)), 3),
            },
        ]
    )
    return SiblingHomogeneity(sib=sib, own=own, other=other, summary=summary)


def wrong_class_closer_fraction(own: list[float], other: list[float]) -> float:
    """Fraction of test questions where the nearest other-class train question
    is more similar than the nearest own-class one."""
    return float(np.mean(np.asarray(other) > np.asarray(own)))


def novelty_calibrated_eval(
    fit: pd.DataFrame,
    val: pd.DataFrame,
    prune_threshold: float,
    k: int = 20,
    vec_kwargs: dict | None = None,
) -> tuple[float, float]:
    """Prune fit questions that are near-siblings of each val question before a
    similarity-weighted k-NN vote, pushing validation novelty toward the test
    level. Returns (validation accuracy, median own-class novelty) at this
    threshold. Declared diagnostic use only — selects no hyperparameter.
    """
    vec = TfidfVectorizer(**(vec_kwargs or {"ngram_range": (1, 1), "sublinear_tf": True}))
    vec.fit(fit.question)
    Xf = vec.transform(fit.question)
    Xq = vec.transform(val.question)
    S = cosine_similarity(Xq, Xf)
    fit_disease = fit.disease.to_numpy()

    correct: list[bool] = []
    novelty: list[float] = []
    for i, gold in enumerate(val.disease):
        s = S[i].copy()
        keep = s < prune_threshold
        s[~keep] = -1
        votes: dict[str, float] = defaultdict(float)
        for j in np.argsort(-s)[:k]:
            if s[j] > 0:
                votes[fit_disease[j]] += s[j]
        correct.append(max(votes, key=votes.get) == gold if votes else False)
        own_sims = S[i][(fit_disease == gold) & keep]
        novelty.append(float(own_sims.max()) if own_sims.size else 0.0)

    return round(float(np.mean(correct)), 3), round(float(np.median(novelty)), 3)
