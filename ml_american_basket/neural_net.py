import numpy as np
import scipy.special as sp

def paramvec_to_params(param, k):
    """
    Map parameter vector to the neural network parameter set.

    Parameters
    ----------
    param : np.array
        (d + 2) * k dimensional float vector
    k : int
        Number of nodes per layer

    Returns
    -------
    (np.array, np.array, np.array)
        Node parameters a,b and c
    """

    num_assets = int((len(param) - 2 * k - 1) / k)
    a = param[:num_assets * k].reshape((num_assets, k))
    b = param[num_assets * k : -(k+1)]
    c = param[-(k+1):]

    return a, b, c


def params_to_paramvec(a, b, c):
    """
    Convenience function to map neural net parameters back to
    parameter vector.

    Parameters
    ----------
    a : np.array
        Linear scale of expit-shift
    b : np.array
        Linear offset of expit shift
    c: np.array
        Scale of expit function

    Returns
    -------
    np.array
        Joined parameter vector.
    """

    return np.concatenate((a.ravel(), b, c))


def neural_net_layer(param, k, x, data_normalization=True):
    """
    Represents single layer of neural network.

    Parameters
    ----------
    param : np.array
        Parameter set of vector a, b and c.
    k : int
        Number of hidden neurons.
    x : np.array
        Prices at timestep t.
    data_normalization : bool, optional
        Indicator whether data should be normalized or not. This
        helps to stabilize the optimization routine. 
        The default is True.

    Returns
    -------
    nn_layer : np.array
        Returns layer response.

    """

    a, b, c = paramvec_to_params(param, k)
    if data_normalization:
        x_norm = (x - x.mean(axis=1))/x.std(axis=1) 
    else:
        x_norm = x
    lin_shifts = np.einsum('ij,il',a, x_norm) + b[:, None]
    nn_layer = np.dot(c[:-1], sp.expit(lin_shifts)) + c[-1]

    return nn_layer


def neural_net_layer_jacobian(param, k, x, y, past_param, l2_reg=1e-5, data_normalization=True):
    """
    Jacobian for quadratic loss function with Tikhonov regularization on past parameter set.
    
    Parameters
    ----------
    param : np.array
        Parameter set of vector a, b and c.
    k : int
        Number of hidden neurons.
    x : np.array
        Prices at timestep t.
    y : np.array
        Strategy value at timestep t.
    past_param : np.array
        Parameter set of vector a, b and c from timestep t+1.
    l2_reg : float
        Scaling parameter for Tikhonov term.
    data_normalization : bool, optional
        Indicator whether data should be normalized or not. This
        helps to stabilize the optimization routine. 
        The default is True.

    Returns
    -------
    loss : float
        The loss value to minimize.

    """
    num_assets = x.shape[0]
    num_params = param.shape[0]
    a, b, c = paramvec_to_params(param, k)

    res = neural_net_layer(param, k, x, data_normalization) - y
    if data_normalization:
        x_norm = (x - x.mean(axis=1))/x.std(axis=1) 
    else:
        x_norm = x
    lin_shifts = np.einsum('ij,il',a, x_norm) + b[:, None]
    expits = sp.expit(lin_shifts)
    expits_prime = sp.expit(-lin_shifts) - sp.expit(-lin_shifts) ** 2

    jacobian_lsq = np.concatenate([np.einsum('ik,jk,k->ijk', 
                                         expits_prime * c[:-1, None], 
                                         x_norm, res).reshape((num_assets * k, x.shape[1])),
                                   expits_prime * c[:-1, None] * res,
                                   expits * res,
                                   res[None, :]])

    return 2 * (np.mean(jacobian_lsq, axis=1) + l2_reg * (param - past_param) / num_params)

