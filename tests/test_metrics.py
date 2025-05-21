import numpy as np
from tspec_metrics_HPC import (
    redundancy_index,
    cluster_entropy,
    change_mag,
    novelty_density,
)

def test_redundancy_index():
    X = np.eye(3, dtype=float)
    assert np.isclose(redundancy_index(X), 1.0)

def test_cluster_entropy():
    X = np.vstack([np.zeros(2), np.zeros(2), np.ones(2), np.ones(2)])
    ce = cluster_entropy(X)
    assert np.isclose(ce, 1.0, atol=1e-5)


def test_change_mag():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert np.isclose(change_mag(a, b), 1.0)


def test_novelty_density():
    Xp = np.eye(2)
    Xn = np.array([[1.0, 0.0], [0.0, 1.0]])
    nd = novelty_density(Xp, Xn, k=2)
    assert np.isclose(nd, 0.0)
