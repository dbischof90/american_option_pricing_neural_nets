import numpy as np

def create_gbm_paths(x0, r, cov, tmin=0, tmax=1, dt=0.01, num_of_samples=100):
    """
    Creates samples of correlated GBMs.

    Parameters
    ----------
    x0 : np.array
        Initial values.
    r : float
        Risk-neutral rate.
    cov : np.array
        Covariance matrix.
    tmin : float, optional
        Time t0. The default is 0.
    tmax : float, optional
        Time t1. The default is 1.
    dt : float, optional
        Time step incremental. The default is 0.01.
    num_of_samples : int, optional
        Number of samples created. The default is 100.

    Returns
    -------
    np.array
        TxMxN array with N GBM paths of T time steps of M instruments.

    """

    m = x0.shape[0]
    tau = int((tmax - tmin)/dt)
    L_vol = np.linalg.cholesky(cov)
    dW = np.random.standard_normal(size=(tau, m, num_of_samples))
    cov_dW = np.einsum('ij,kjl->kil', L_vol, dW)
    drift = np.outer(np.arange(tmin, tmax, dt), r - np.diag(cov) / 2)

    return np.einsum('j,ijl->ijl', x0, np.exp(drift[:, :, None] + dt ** .5 * np.cumsum(cov_dW, axis=0)))
