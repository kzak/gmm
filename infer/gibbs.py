from pdb import set_trace

import numpy as np
from scipy import stats


def gmm_gen(N, D, K, alpha, nu, W, seed=None):
    """
    Args
        N: Int > 0
        D: Int > 0
        K: Int > 0
        alpha: K-array
        nu: Int > D - 1
        W: (D, D) matrix
    Returns
        X: (N, D) matrix
    """
    pi = stats.dirichlet(alpha, seed).rvs()[0]
    Lmd = [stats.wishart(df=nu, scale=W).rvs() for _ in range(K)]
    mu = [stats.uniform(scale=3).rvs(D) for _ in range(K)]

    X = []
    for n in range(N):
        s_n = np.random.multinomial(1, pvals=pi)
        for k in range(K):
            if s_n[k] == 1:
                X += [stats.multivariate_normal(mu[k], Lmd[k]).rvs()]
                break

    return np.array(X).reshape(N, D)


def gibbs_init(X, K):
    """
    Args
        X: (N, D) matrix
            Item matrix
        K: Int > 0
            Num of class (mixture)
    Returns
        smpl_dict: Dict
            Initilized memory to save gibbs sampled items.
            {
                "pi": [[pi_1, ..., pi_K], ...],
                "mu": [[mu_1, ..., mu_K], ...],
                "Lmd": [[Lmd_1, ..., Lmd_K], ...],
                "S": [[S], ...]
            }
            pi_k: Double in [0, 1]
                Probability of belonging to each class
            mu_k: d-dim vector
                Mean vector of each class
            Lmd_k: (d, d) matrix
                Scale matrix of each class
            S: (N, K) matrix
                s_nk in {0, 1} and \sum_k s_nk = 1
    """
    # Initialize
    smpl_dict = {}

    # Initialize pi
    smpl_dict["pi"] = [[1 / K for _ in range(K)]]

    # Initialize mu
    mu0 = X.mean(axis=0)
    smpl_dict["mu"] = [[mu0 for _ in range(K)]]

    # Initialize Lmd
    Lmd0 = np.corrcoef(X.T)
    smpl_dict["Lmd"] = [[Lmd0 for _ in range(K)]]

    # Init S: (N, K) matrix
    S = [stats.bernoulli(smpl_dict["pi"][-1][k]).rvs(len(X)) for k in range(K)]
    smpl_dict["S"] = [np.array(S).reshape(len(X), -1)]

    return smpl_dict


if __name__ == "__main__":
    pass
