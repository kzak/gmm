from pdb import set_trace

import numpy as np
from numpy.linalg import pinv
from scipy import stats


def gibbs_init(X, K):
    """
    Args
        X: (N, D) matrix
            Items
        K: Int > 0
            Num of classes (mixture)
    Returns
        smpl_hist: Dict
           History of gibbs sampling which has latents variables and parameters.
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
    N, D = X.shape
    eps = 1.0e-2

    smpl_hist = {}

    # Initialize pi
    smpl_hist["pi"] = [[1 / K for _ in range(K)]]

    # Initialize mu
    mu0 = X.mean(axis=0)
    smpl_hist["mu"] = [
        [
            mu0.copy() + np.random.multivariate_normal(np.zeros(D), eps * np.eye(D))
            for _ in range(K)
        ]
    ]

    # Initialize Lmd
    Lmd0 = np.corrcoef(X.T)
    smpl_hist["Lmd"] = [[Lmd0 for _ in range(K)]]

    # Init S: (N, K) matrix
    S = [np.random.multinomial(1, pvals=smpl_hist["pi"][-1]) for _ in range(len(X))]
    smpl_hist["S"] = [np.array(S).reshape(len(X), -1)]

    return smpl_hist


def draw_s_n(x_n, mu, Lmd, pi, eps=1.0e-5):
    eta_n = []
    for k in range(len(mu)):
        eta_nk = np.exp(
            -0.5 * (x_n - mu[k]) @ Lmd[k] @ (x_n - mu[k]).T
            + 0.5 * np.linalg.slogdet(Lmd[k])[1]
            + np.log(pi[k])
        )
        eta_n += [eta_nk + eps]  # eps ; to avoid zero division

    eta_n = np.array(eta_n) / sum(eta_n)

    return np.random.multinomial(1, pvals=eta_n)


def draw_Lmd_k_mu_k(X, S_k, m, beta, nu, W_inv):
    X_k = X[S_k == 1]

    sum_s_k = S_k.sum()
    beta_hat = sum_s_k + beta
    m_hat_k = (X_k.sum(axis=0) + beta * m) / beta_hat
    nu_hat = sum_s_k + nu
    sum_xx_n = sum([np.outer(X_k[n], X_k[n]) for n in range(len(X_k))])

    W_hat_k_inv = (
        sum_xx_n + beta * np.outer(m, m) - beta_hat * np.outer(m_hat_k, m_hat_k) + W_inv
    )

    Lmd_k = stats.wishart(nu_hat, pinv(W_hat_k_inv)).rvs()
    mu_k = stats.multivariate_normal(m_hat_k, pinv(beta_hat * Lmd_k)).rvs()

    return Lmd_k, mu_k


def draw_pi(S, alpha):
    alpha_hat = np.sum(S.T, axis=1) + alpha
    return stats.dirichlet.rvs(alpha_hat)[0]


def gibbs_sampling(X, K, n_iter=100):
    """
    Args
        X: (N, D) matrix
        K: Int > 0
        n_iter: Int > 0
    Returns
        smpl_hist: Dict
            {
                "pi": [[pi_1, ..., pi_K], ...],
                "mu": [[mu_1, ..., mu_K], ...],
                "Lmd": [[Lmd_1, ..., Lmd_K], ...],
                "S": [[S], ...]
            }
    """
    # Initialization
    N, D = X.shape
    ss = gibbs_init(X, K)

    # Hyper parameters
    alpha = np.ones(K)
    beta = 1.0
    nu = D
    W_inv = np.eye(D)

    # Gibbs sampling
    for i in range(n_iter):

        # Draw S
        S = []
        for n in range(N):
            S += [draw_s_n(X[n], ss["mu"][-1], ss["Lmd"][-1], ss["pi"][-1])]
        ss["S"] += [np.array(S).reshape(N, K)]

        # Draw Lmd, mu
        Lmd = []
        mu = []
        for k in range(K):
            Lmd_k, mu_k = draw_Lmd_k_mu_k(
                X,
                ss["S"][-1][:, k],  # S_k
                ss["mu"][-1][k],  # m = mu_k
                beta,
                nu,
                W_inv,
            )
            Lmd += [Lmd_k]
            mu += [mu_k]

        ss["Lmd"] += [Lmd]
        ss["mu"] += [mu]

        # Draw pi
        pi = draw_pi(ss["S"][-1], alpha)
        ss["pi"] += [pi]

    return ss


if __name__ == "__main__":
    pass
