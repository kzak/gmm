import numpy as np
from scipy import stats


def load_data(N, D, K):
    """
    Args
        N: Int > 0
            Num of items
        D: Int > 0
            Dim of the item
        K: Int > 0
            Num of classes / mixtures
    Returns
        X: (N, D) matrix
            Items
        C: N array
            Class numbers of items
    """
    alpha = np.ones(K)
    nu = D
    W = np.eye(D)

    pi = stats.dirichlet(alpha).rvs()[0]

    mu = [stats.uniform(scale=5).rvs(D) for _ in range(K)]
    Lmd = [stats.wishart(df=nu, scale=W).rvs() for _ in range(K)]

    X = []
    C = []
    for n in range(N):
        s_n = np.random.multinomial(1, pvals=pi)
        for k in range(K):
            if s_n[k] == 1:
                X += [stats.multivariate_normal(mu[k], Lmd[k]).rvs()]
                C += [k]
                break

    X = np.array(X).reshape(N, D)
    C = np.array(C)

    return X, C
