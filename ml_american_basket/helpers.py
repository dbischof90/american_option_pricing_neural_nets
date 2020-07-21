import numpy as np
import numbers
import collections.abc

def create_paths(x0, r, cov, tmin=0, tmax=1, dt=0.01, num_of_samples=100):

    m = x0.shape[0]
    tau = int((tmax - tmin)/dt)

    L_vol = np.linalg.cholesky(cov)
    dW = dt ** 0.5 * np.random.standard_normal(size=(tau - 1, m, num_of_samples))
    cov_dW = np.einsum('ij,kjl->kil', L_vol, np.insert(dW, 0, 0, axis=0))
    drift = np.outer(np.arange(tmin, tmax, dt), r - np.diag(cov) / 2)

    return np.einsum('j,ijl->ijl', x0, np.exp(drift[:, :, None] + np.cumsum(cov_dW, axis=0)))
