import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from metric_utils import redundancy_index, cluster_entropy

def test_redundancy_index():
    X = np.eye(3, dtype=float)
    assert np.isclose(redundancy_index(X), 1.0)

def test_cluster_entropy():
    X = np.vstack([np.zeros(2), np.zeros(2), np.ones(2), np.ones(2)])
    ce = cluster_entropy(X)
    assert np.isclose(ce, 1.0, atol=1e-5)

