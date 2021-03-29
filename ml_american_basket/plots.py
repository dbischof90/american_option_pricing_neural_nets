import matplotlib.pyplot as plt
import neural_net as nn

def plot_sample_vs_strategy_at_t(nn_parameter, k, sample, cont_or_execute, t, payoff_name, folder='.', **kwargs):
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot()
    ax.scatter(sample, cont_or_execute, label="Simulated execution strategy values")
    ax.scatter(sample, nn.neural_net_layer(nn_parameter, k, sample), label="Calibrated neural net response")
    ax.legend()
    ax.set_title(f"Continuation values versus model calibration at time $t = {t:.3f}$ for {payoff_name} option")
    ax.set_xlabel("Underlying price")
    ax.set_ylabel("Strategy value")
    fig.savefig(f"./{folder}/" + f"{payoff_name}_{t:.3f}_scatter_cv".replace('.', '_') + '.png')
    plt.close(fig)
