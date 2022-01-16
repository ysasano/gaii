def test_data1():
    A = np.array([[0.03, 0.8], [0.8, 0.03]])
    Sigma = np.array([[0.2, 0.1], [0.1, 0.2]])
    initiali_state = np.random.multivariate_normal(
        [0, 0], [[0.1, 0], [0, 0.1]], size=1
    )[0]
    return var.generate(initiali_state, A, Sigma, N=1000)


def test_data1():
    A = np.array([[0.03, 0.8], [0.8, 0.03]])
    Sigma = np.array([[0.2, 0.1], [0.1, 0.2]])
    initiali_state = np.random.multivariate_normal(
        [0, 0], [[0.1, 0], [0, 0.1]], size=1
    )[0]
    return var.generate(initiali_state, A, Sigma, N=1000)


def test_data1():
    A = np.array([[0.03, 0.8], [0.8, 0.03]])
    Sigma = np.array([[0.2, 0.1], [0.1, 0.2]])
    initiali_state = np.random.multivariate_normal(
        [0, 0], [[0.1, 0], [0, 0.1]], size=1
    )[0]
    return var.generate(initiali_state, A, Sigma, N=1000)
