"""Arm 1 retrieval core: TF-IDF vectorise the class-evidence index, k-NN by
cosine similarity, similarity-weighted vote over neighbour labels. Also holds
the supervised TF-IDF baselines (LinearSVC / LogisticRegression /
RandomForest) compared against retrieval in the same arm.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC

from amlh.config import SEED
from amlh.features import build_vectoriser


def knn_rank(
    index_texts: list[str],
    index_labels: list[str],
    query_texts: list[str],
    k: int,
    vec_kwargs: dict,
) -> tuple[list[list[str]], list[float]]:
    """Fit TF-IDF on `index_texts` only, transform `query_texts`, and rank
    labels per query by summed cosine similarity over the k nearest
    neighbours. Returns (ranked_labels_per_query, top_similarity_per_query).
    """
    vec = build_vectoriser(**vec_kwargs)
    X_index = vec.fit_transform(index_texts)
    X_query = vec.transform(query_texts)

    nn = NearestNeighbors(n_neighbors=min(k, len(index_texts)), metric="cosine", algorithm="brute")
    nn.fit(X_index)
    distances, neighbour_idx = nn.kneighbors(X_query)
    similarities = 1.0 - distances

    labels = np.asarray(index_labels)
    ranked: list[list[str]] = []
    top_sim: list[float] = []
    for row_sims, row_idx in zip(similarities, neighbour_idx):
        vote: dict[str, float] = {}
        for lab, sim in zip(labels[row_idx], row_sims):
            vote[lab] = vote.get(lab, 0.0) + sim
        ranked.append(sorted(vote, key=vote.get, reverse=True))
        top_sim.append(float(row_sims[0]))

    return ranked, top_sim


_MODELS = {
    "svc": lambda seed: LinearSVC(random_state=seed),
    "logreg": lambda seed: LogisticRegression(random_state=seed, max_iter=1000),
    "random_forest": lambda seed: RandomForestClassifier(random_state=seed),
}


def linear_rank(
    train_texts: list[str],
    train_labels: list[str],
    query_texts: list[str],
    vec_kwargs: dict,
    model: str,
    seed: int = SEED,
) -> tuple[list[list[str]], list[float]]:
    """Fit TF-IDF + a supervised classifier on `train_texts`/`train_labels`
    only, rank classes per query by decision score (predict_proba where
    available, else decision_function) descending. `model` is one of
    "svc", "logreg", "random_forest".
    """
    if model not in _MODELS:
        raise ValueError(f"model must be one of {sorted(_MODELS)}, got {model!r}")

    vec = build_vectoriser(**vec_kwargs)
    X_train = vec.fit_transform(train_texts)
    X_query = vec.transform(query_texts)

    clf = _MODELS[model](seed)
    clf.fit(X_train, train_labels)

    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_query)
    else:
        scores = clf.decision_function(X_query)

    classes = clf.classes_
    ranked: list[list[str]] = []
    top_score: list[float] = []
    for row in scores:
        order = np.argsort(row)[::-1]
        ranked.append([classes[i] for i in order])
        top_score.append(float(row[order[0]]))

    return ranked, top_score
