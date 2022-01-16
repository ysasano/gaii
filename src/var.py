import numpy as np

from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import AR


def generate(initiali_state, A, Sigma, N=1000):
    state = initiali_state
    state_list = [state]
    for _ in range(N):
        E = np.random.multivariate_normal(
            np.zeros(initiali_state.shape), Sigma, size=1
        )[0]
        state = list(A @ np.array(state).T + E.T)
        state_list.append(state)
    return np.array(state_list)


def reestimate_var(state_list):
    ar = VAR(state_list).fit(trend="n", maxlags=1)
    return {"A": ar.params, "SigmaE": ar.sigma_u}


def reestimate_var2(state_list):
    ar0 = AR(state_list[:, 0]).fit(trend="n", maxlag=1)
    ar1 = AR(state_list[:, 1]).fit(trend="n", maxlag=1)
    return {
        "A": np.array([[ar0.params[0], 0], [0, ar1.params[0]]]),
        "SigmaE": np.array([[ar0.sigma2, 0], [0, ar0.sigma2]]),
    }
