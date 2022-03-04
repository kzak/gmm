import numpy as np
from infer.gibbs import gibbs_init, gmm_gen
from scipy import stats


def test_gmm_gen():
    N = 10
    D = 2
    K = 3
    alpha = [1, 1, 1]
    nu = 3
    W = np.array([[1, 0], [0, 1]])

    X = gmm_gen(N, D, K, alpha, nu, W, seed=None)

    assert (N, D) == X.shape


def test_gibbs_init():
    N = 100
    D = 2
    K = 2
    alpha = [1, 1]
    nu = 3
    W = np.array([[1, 0], [0, 1]])
    X = gmm_gen(N, D, K, alpha, nu, W)

    ss = gibbs_init(X, K)

    assert K == len(ss["pi"][0])
    assert X.ndim == len(ss["mu"][0])
    assert (X.ndim, X.ndim) == ss["Lmd"][0][0].shape
    assert (len(X), X.ndim) == ss["S"][0].shape
