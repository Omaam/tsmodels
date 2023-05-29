"""Example main.
"""
import matplotlib.pyplot as plt
import numpy as np

from tsmodels.result import var
import dataset


def plot_impulse_response(var_analyzer):
    num_series = var_analyzer.num_series
    num_lagsteps = 100
    irf = var_analyzer.compute_impulse_response(
        num_lagsteps, orthogonal=False)
    fig, ax = plt.subplots(num_series, num_series,
                           sharex=True, sharey=True)
    lags = np.arange(1, num_lagsteps+1, 1)
    for i in range(num_series):
        for j in range(num_series):
            ax[i, j].plot(lags, irf[:, i, j], color="k")
            ax[i, j].axhline(0, color="r")
    plt.tight_layout()
    plt.show()


def main():
    arcoef = dataset.get_arcoef()
    noise_cov = dataset.get_noise_covariance()

    # Change matrices in order to be recursive structure.
    arcoef[..., :, :] = arcoef[..., ::-1, ::-1]
    noise_cov[..., :, :] = noise_cov[..., ::-1, ::-1]

    var_analyzer = var.VarAnalyzer(arcoef, noise_cov)
    plot_impulse_response(var_analyzer)


if __name__ == "__main__":
    main()
