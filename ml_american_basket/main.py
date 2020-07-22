import math as m
import numpy as np
import scipy.optimize as opt
import functools as ft
import matplotlib.pyplot as plt

def create_paths(x0, r, cov, tmin=0, tmax=1, dt=0.01, num_of_samples=100):
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


def log_squasher(x):
    return 1/(1 + np.exp(-x))


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
        Indicator whether data should be normalizedor not. This
        can help to stabilize the optimization. The default is True.

    Returns
    -------
    nn_layer : np.array
        Returns layer response.

    """
    a = param[:x.shape[0] * k].reshape((x.shape[0], k))
    b = param[x.shape[0] * k : -(k+1)]
    c = param[-(k+1):]
    if data_normalization:
        x_hat = (x - x.mean())/x.std()
    else:
        x_hat = x
    lin_shifts = np.einsum('ij,il',a, x_hat) + b[:, None]
    nn_layer = np.dot(c[:-1], log_squasher(lin_shifts)) + c[-1]
    return nn_layer

def quadratic_loss(param, model, data, k, penalty_type):
    """
    Generalized quadratic loss function. Can take three forms to regularize 
    parameters for unwanted behavior.

    Parameters
    ----------
    param : np.array
        Parameter set of vector a, b and c.
    model : function
        Regression function to model.
    data : np.array
        Observations to regress against.
    k : int
        Number of hidden neurons.
    penalty_type : string
        Determines the type of penalty.

    Raises
    ------
    ValueError
        Needs specified penalty types.

    Returns
    -------
    loss : float
        The loss valu to minimize.

    """
    a_b = param[:data.shape[0] * k + k]
    c = param[-(k+1):-1]
    
    sq_deviations = np.sum((model(param) - data) ** 2)
    if penalty_type == 'simple':
        loss = sq_deviations
    elif penalty_type == 'restr_lin_shift':
        loss = sq_deviations + np.maximum(np.abs(a_b) - 5, 0).sum() ** 10
    elif penalty_type == 'restr_conodals':
        loss = sq_deviations 
    elif penalty_type == 'full':
        loss = sq_deviations\
            + np.maximum(np.abs(a_b) - 5, 0).sum() ** 10\
            + 5 * max((abs(c.sum()/np.abs(c).sum()) - 0.8, 0)) * max(abs(c)) ** 2
    else:
        raise ValueError("Unknown penalty function type")
    
    return loss
           

def infer_continuation_value(price_paths, k, payoff_dict):
    """
    Continuation inference routine. Builds up neural network and derives 
    parameters back to front to compute the final continuation values.

    Parameters
    ----------
    price_paths : np.array
        Simulated price paths.
    k : int
        Number of hidden neurons per layer.
    payoff_dict : dict
        Dictionary with payoff informations.

    Returns
    -------
    cont_parameters : np.array
        Returns the continuation values of the option.

    """
    # Initialize continuation value at t = T
    cont_values = [0]    
    
    # Sets payoff information
    payoff = payoff_dict['func']
    tmax = payoff_dict['tmax']
    tmin = payoff_dict['tmin']
    strike = payoff_dict['strike']
    dt = (tmax - tmin)/price_paths.shape[0]
    
    closest_payoff = payoff(price_paths[-1], strike, tmax)
    next_continuation = 0
    cont_init = np.ones(shape=(price_paths.shape[1] * k + 2*k + 1))
    
    # Build continuation samples by iterating backwards through time
    for x_t, t in zip(price_paths[-2::-1], np.arange(tmax - dt, tmin, -dt)):
        parametrized_net_layer = ft.partial(neural_net_layer, k=k, x=x_t)
        cont_or_execute = np.maximum(next_continuation, closest_payoff)
        loss_func = ft.partial(quadratic_loss, model=parametrized_net_layer,
                                               data=cont_or_execute,
                                               k=k,
                                               penalty_type='full')
        cont_result = opt.minimize(loss_func, x0=cont_init, 
                                              method='BFGS',
                                              options={'gtol': 1e-3})
        
        cont_init = cont_result.x
        
        closest_payoff = payoff(x_t, strike, t)
        next_continuation = parametrized_net_layer(cont_init)
        
        cont_values.append(next_continuation.mean())
    
    return np.array(cont_values)[::-1]


def vanilla_put(x, K, tau, r):
    return m.exp(-r*tau) * np.maximum(K - x, 0)

def vanilla_call(x, K, tau, r):
    return m.exp(-r*tau) * np.maximum(x - K, 0)


if __name__ == '__main__':
    
    x0 = np.array([100])
    cov = np.array([[0.25]]) ** 2
    r = 0.05
    k = 2
    dt = 1/50
    
    payoff_dict = {'func': ft.partial(vanilla_call, r=r),
                   'tmax': 1,
                   'tmin': 0,
                   'strike': 90}
    
    price_paths = create_paths(x0, r, cov, num_of_samples=10000, dt=dt)
    cont_values = infer_continuation_value(price_paths, k, payoff_dict)
    
    # Option value: Execute either directly or hold the option.
    V_0 = max((cont_values[0], payoff_dict['func'](x0, payoff_dict['strike'], 0)))
    
    plt.plot(np.arange(payoff_dict['tmin'], payoff_dict['tmax'], dt),
             cont_values)
    