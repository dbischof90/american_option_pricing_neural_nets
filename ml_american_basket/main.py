import numpy as np
import scipy.optimize as opt
import functools as ft

def log_squasher(x):
    return 1/(1 + np.exp(-x))

def neural_net_layer(param, k, x):
    a, b, c = param
    lin_shift = np.dot(a, x) + b
    if k == 0:
        nn_layer = c[0]
    else:
        nn_layer = c[:-1].dot(log_squasher(x)) + c[-1]
    return nn_layer

def quadratic_loss(param, model, data):
    return np.sum( (model(param) - data) ** 2 )

def infer_continuation_value(price_paths, payoff, strike):
    # Initialize return and the continuation value at t = T
    q = [np.zeros(price_paths.shape[1:])]

    # Build continuation samples
    for x in price_paths[::-1]:
        q_t = np.maximum(q[-1], payoff(x, strike))
        q.append(q_t)
    
    return q

def neural_network_regression(price_paths, continuation_value, ls_size):

    results = []
    for k in 2 ** np.arange(1, 5, 1, dtype=int):
        # Calibrate neural net with given node complexity k
        parametrized_net_layer = ft.partial(neural_net_layer, k=k, x=price_paths)
        loss_func = ft.partial(quadratic_loss, model=parametrized_net_layer,
                                               data=continuation_value)
        res = opt.minimize(loss_func, x0=0)
        results.append(res.x)

    return results
