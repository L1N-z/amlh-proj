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
    depth: int | None = None,
) -> tuple[list[list[str]], list[float]]:
    """Fit TF-IDF on `index_texts` only, transform `query_texts`, and rank
    labels per query by summed cosine similarity over the k nearest
    neighbours. Returns (ranked_labels_per_query, top_similarity_per_query).

    `depth`, if given, retrieves `min(max(k, depth), len(index_texts))`
    neighbours instead of just `k`. Without it, `ranked` has exactly as many
    entries as there are distinct labels among the k nearest neighbours — at
    the frozen `k_neighbors=1` that is always length 1, so there is no top-5
    or top-20 to score Top-5/MRR against and no shortlist for Arm 3 to select
    from. The head of the ranking is always built from `row_idx[:k]` /
    `row_sims[:k]` — the identical slice used when `depth=None` — so passing
    `depth` can only append below the existing head, never reorder it;
    `depth=None` (every caller that doesn't pass it) is untouched. Labels
    first seen among neighbours k+1..depth are appended after the head,
    ordered by their best similarity in that range, deduplicated against the
    head.
    """
    n_retrieve = min(k, len(index_texts))
    if depth is not None:
        n_retrieve = min(max(k, depth), len(index_texts))

    vec = build_vectoriser(**vec_kwargs)
    X_index = vec.fit_transform(index_texts)
    X_query = vec.transform(query_texts)

    nn = NearestNeighbors(n_neighbors=n_retrieve, metric="cosine", algorithm="brute")
    nn.fit(X_index)
    distances, neighbour_idx = nn.kneighbors(X_query)
    similarities = 1.0 - distances

    labels = np.asarray(index_labels)
    ranked: list[list[str]] = []
    top_sim: list[float] = []
    for row_sims, row_idx in zip(similarities, neighbour_idx):
        vote: dict[str, float] = {}
        for lab, sim in zip(labels[row_idx[:k]], row_sims[:k]):
            vote[lab] = vote.get(lab, 0.0) + sim
        head = sorted(vote, key=vote.get, reverse=True)

        if depth is not None:
            tail_sim: dict[str, float] = {}
            for lab, sim in zip(labels[row_idx[k:]], row_sims[k:]):
                if lab in vote:
                    continue
                if sim > tail_sim.get(lab, -1.0):
                    tail_sim[lab] = sim
            head = head + sorted(tail_sim, key=tail_sim.get, reverse=True)

        ranked.append(head)
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
