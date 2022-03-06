from data_gen.gmm_data_gen import load_data


def test_load_data():
    N = 10
    D = 2
    K = 3

    X, C = load_data(N, D, K)

    assert (N, D) == X.shape
    assert (N,) == C.shape
