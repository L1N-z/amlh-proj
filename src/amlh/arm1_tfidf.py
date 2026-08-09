"""Arm 1 retrieval core: TF-IDF vectorise the class-evidence index, k-NN by
cosine similarity, similarity-weighted vote over neighbour labels.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

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
