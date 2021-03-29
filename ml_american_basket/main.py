import numpy as np
import functools as ft
import matplotlib.pyplot as plt

import payoffs as po
import calibration as cal
import neural_net as nn
import data_generation as dg
import plots

if __name__ == '__main__':
    
    x0 = np.array([100])
    cov = np.array([[0.25]]) ** 2
    r = 0.05
    k = 2
    dt = 1/12
    tmin = 0
    tmax = 1

    payoff_dict = {'func': ft.partial(po.vanilla_put, r=r, K=90),
                   'tmax': tmax,
                   'tmin': tmin}
    
    opt_values = []
    for i in range(100):
        print(i)
        price_paths = dg.create_gbm_paths(x0, r, cov, tmin, tmax, dt, 1000)
        nn_parameters = cal.infer_continuation_value_by_neural_net(price_paths, k, payoff_dict, 
                                                                   callback=plots.plot_sample_vs_strategy_at_t, 
                                                                   payoff_name='Vanilla put', folder='Plots') 
        # Option value: Execute either directly or hold the option.
        V_0 = max(nn.neural_net_layer(nn_parameters[0], k, x0[:, None], data_normalization=False)[0], 
                  payoff_dict['func'](x0, 0))
        opt_values.append(V_0)

    plt.boxplot(opt_values)
    plt.show() 

