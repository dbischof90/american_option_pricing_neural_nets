import numpy as np
import scipy.optimize as opt
import neural_net as nn

def quadratic_loss(param, k, x, y, past_param, l2_reg=1e-5, data_normalization=True):
    """
    Quadratic loss function with Tikhonov regularization on past parameter set.
    

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
    l2_error = np.mean((nn.neural_net_layer(param, k, x, data_normalization) - y) ** 2)
    ridge_term = np.mean((param - past_param) ** 2)
    
    return l2_error + l2_reg * ridge_term
 

def infer_continuation_value_by_neural_net(price_paths, k, payoff_dict, verbose=False, callback=None, **kwargs):
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
    nn_parameters : np.array
        Returns the parameters of the neural net regression.

    """

    nn_parameter_list = []    
    use_jac = bool(kwargs['use_jac']) if 'use_jac' in kwargs else False
    l2_reg_val = float(kwargs['l2_reg']) if 'l2_reg' in kwargs else 1e-5

    # Sets payoff information
    payoff = payoff_dict['func']
    tmax = payoff_dict['tmax']
    tmin = payoff_dict['tmin']
    dt = (tmax - tmin)/price_paths.shape[0]
    
    # Initialization: Payoff at terminal time, zero continuation value,
    # initial parameter estimates and optimal strategy
    execute_at_t = payoff(price_paths[-1], tmax - tmin)
    continuation_at_t = np.zeros_like(price_paths[-1][0])
    cont_or_execute = np.maximum(continuation_at_t, execute_at_t)
    nn_parameter = np.zeros(shape=((price_paths.shape[1] + 2) * k + 1))
    
    # Iterate backwards through sample set
    for sample, t in zip(price_paths[-2::-1], np.arange(tmax - dt, tmin, -dt)):

        calibration_result = opt.minimize(quadratic_loss, 
                                          x0=nn_parameter,
                                          jac=nn.neural_net_layer_jacobian if use_jac else None,
                                          args=(k, sample, cont_or_execute, nn_parameter, l2_reg_val),
                                          method='BFGS',
                                          )

        # If optimization was successfull, save the new parameter set
        if verbose:
            if not calibration_result.success:
                if calibration_result.status == 2:
                    print(f"Warning at t = {t:.3f}: {calibration_result.message}")
                    print(f"Range of Jacobian: [{calibration_result.jac.min()}, {calibration_result.jac.max()}]")
                else:
                    print(f"Problem at t = {t:.3f}: {calibration_result.message}")         
        nn_parameter = calibration_result.x
        
        if callback is not None:
            callback(nn_parameter, k, sample, cont_or_execute, t, **kwargs)

        # Determine payoff and evaluate neural net estimate of conditional expectation
        execute_at_t = payoff(sample, t - tmin)
        continuation_at_t = nn.neural_net_layer(nn_parameter, k, sample)

        # Evaluate next execution strategy per path calibrate neural net to the sample outcomes
        cont_or_execute = np.maximum(continuation_at_t, execute_at_t)
    
        # Scale parameters back to standard space
        mu, sigma = sample.mean(axis=1), sample.std(axis=1)
        a, b, c = nn.paramvec_to_params(nn_parameter.copy(), k)
        a /= sigma[:, None]
        b -= (a * mu[:, None]).sum(axis=0)
        rescaled_nn_parameter = nn.params_to_paramvec(a, b, c)

        # Save parameters
        nn_parameter_list.append(rescaled_nn_parameter)
    
    return np.array(nn_parameter_list)[::-1]

