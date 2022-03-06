import numpy as np
from data_gen.gmm_data_gen import load_data
from infer.gibbs import gibbs_init, gibbs_sampling
from scipy import stats


def test_gibbs_init():
    N = 100
    D = 2
    K = 2
    X, _ = load_data(N, D, K)

    ss = gibbs_init(X, K)

    assert K == len(ss["pi"][0])
    assert X.ndim == len(ss["mu"][0])
    assert (X.ndim, X.ndim) == ss["Lmd"][0][0].shape
    assert (len(X), X.ndim) == ss["S"][0].shape


def test_gibbs_sampling():
    N = 100
    D = 2
    K = 2
    X, _ = load_data(N, D, K)

    n_iter = 10
    ss = gibbs_sampling(X, K, n_iter)

    assert (n_iter + 1) == len(ss["pi"])
