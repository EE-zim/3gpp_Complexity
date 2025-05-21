"""Common metric functions used by the 3GPP complexity scripts."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import util
from sklearn.cluster import KMeans
from scipy.stats import entropy

__all__ = [
    "semantic_spread",
    "redundancy_index",
    "cluster_entropy",
    "change_mag",
    "novelty_density",
]

def semantic_spread(X: np.ndarray) -> float:
    """Return the total variance of the embedding matrix."""
    return float(np.trace(np.cov(X, rowvar=False)))

def redundancy_index(X: np.ndarray, k: int = 1000) -> float:
    """Measure redundancy via average pairwise cosine similarity.

    A subset of ``k`` vectors is used if ``len(X)`` exceeds ``k``.
    """
    if len(X) > k:
        X = X[np.random.choice(len(X), k, replace=False)]
    sims = util.cos_sim(X, X).cpu().numpy()
    return 1.0 - float(sims[np.triu_indices_from(sims, 1)].mean())

def cluster_entropy(X: np.ndarray, sample: int = 5000) -> float:
    """Entropy of KMeans clusters drawn from ``X``."""
    if len(X) > sample:
        X = X[np.random.choice(len(X), sample, replace=False)]
    labels = KMeans(n_clusters=int(np.sqrt(len(X))), n_init="auto", random_state=0).fit_predict(X)
    p = np.bincount(labels) / len(labels)
    return float(entropy(p, base=2))

def change_mag(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between mean embeddings of two releases."""
    return 1.0 - float(util.cos_sim(a, b))

def novelty_density(Xp: np.ndarray, Xn: np.ndarray, k: int = 2000) -> float:
    """Proportion of novel sentences in ``Xn`` relative to pool ``Xp``."""
    if len(Xn) > k:
        Xn = Xn[np.random.choice(len(Xn), k, replace=False)]
    sims = util.cos_sim(Xn, Xp).cpu().numpy()
    return float((1.0 - sims.max(1)).mean())

