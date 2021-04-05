import numpy as np
import functools as ft
import pathlib

import payoffs as po
import calibration as cal
import neural_net as nn
import data_generation as dg
import plots


def boxplot_for_put_option():
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
    
    # Generate price samples
    option_values = []
    for i in range(1, 101):
        print(f"Iteration {i}/100")
        price_paths = dg.create_gbm_paths(x0, r, cov, tmin, tmax, dt, 1000)
        nn_parameters = cal.infer_continuation_value_by_neural_net(price_paths, k, payoff_dict, verbose=True) 
        # Option value: Execute either directly or hold the option.
        V_0 = max(nn.neural_net_layer(nn_parameters[0], k, x0[:, None], data_normalization=False)[0], 
                  payoff_dict['func'](x0, 0))
        option_values.append(V_0)

    plots.create_price_boxplot(option_values, "Vanilla put", "Plots")


def boxplot_for_straddle_spread_option():
    x0 = np.array([100])
    cov = np.array([[0.5]]) ** 2
    r = 0.05
    k = 4
    dt = 1/12
    tmin = 0
    tmax = 1

    payoff_dict = {'func': ft.partial(po.strangle_spread, r=r, K1=50, K2=90, K3=110, K4=150),
                   'tmax': tmax,
                   'tmin': tmin}
    
    # Generate price samples
    option_values = []
    for i in range(1, 101):
        print(f"Iteration {i}/100")
        price_paths = dg.create_gbm_paths(x0, r, cov, tmin, tmax, dt, 1000)
        nn_parameters = cal.infer_continuation_value_by_neural_net(price_paths, k, payoff_dict) 
        # Option value: Execute either directly or hold the option.
        V_0 = max(nn.neural_net_layer(nn_parameters[0], k, x0[:, None], data_normalization=False)[0], 
                  payoff_dict['func'](x0, 0))
        option_values.append(V_0)

    plots.create_price_boxplot(option_values, "Straddle spread", "Plots")


def boxplot_for_basket_straddle_spread_option():
    x0 = np.array([100, 100, 100, 100, 100])
    vol = np.array([[0.3024, 0.1354, 0.0722, 0.1367, 0.1641], 
                    [0.1354, 0.2270, 0.0613, 0.1264, 0.1610], 
                    [0.0722, 0.0613, 0.0717, 0.0884, 0.0699],
                    [0.1367, 0.1264, 0.0884, 0.2937, 0.1394],
                    [0.1641, 0.1610, 0.0699, 0.1394, 0.2535]])
    cov = vol @ vol
    r = 0.05
    k = 12
    dt = 1/48
    tmin = 0
    tmax = 1

    payoff_dict = {'func': ft.partial(po.mean_strangle_spread, r=r, K1=75, K2=90, K3=110, K4=125),
                   'tmax': tmax,
                   'tmin': tmin}
    
    # Generate price samples
    option_values = []
    for i in range(1, 101):
        print(f"Iteration {i}/100")
        price_paths = dg.create_gbm_paths(x0, r, cov, tmin, tmax, dt, 1000)
        nn_parameters = cal.infer_continuation_value_by_neural_net(price_paths, k, payoff_dict) 
        # Option value: Execute either directly or hold the option.
        V_0 = max(nn.neural_net_layer(nn_parameters[0], k, x0[:, None], data_normalization=False)[0], 
                  payoff_dict['func'](x0, 0))
        option_values.append(V_0)

    plots.create_price_boxplot(option_values, "Basket straddle spread", "Plots")


def continuation_value_plots_straddle_spread():
    x0 = np.array([100])
    cov = np.array([[0.5]]) ** 2
    r = 0.05
    k = 4
    dt = 1/12
    tmin = 0
    tmax = 1

    payoff_dict = {'func': ft.partial(po.strangle_spread, r=r, K1=50, K2=90, K3=110, K4=150),
                   'tmax': tmax,
                   'tmin': tmin}

    price_paths = dg.create_gbm_paths(x0, r, cov, tmin, tmax, dt, 1000)
    _ = cal.infer_continuation_value_by_neural_net(price_paths, k, payoff_dict, verbose=True,
                                                   callback=plots.plot_sample_vs_strategy_at_t, 
                                                   payoff_name='Straddle spread', folder='Plots') 


if __name__ == '__main__':
    plotpath = pathlib.Path("Plots")
    plotpath.mkdir(exist_ok=True)

    boxplot_for_put_option()
