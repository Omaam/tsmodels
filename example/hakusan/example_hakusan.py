"""VAR model example.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsmodels.result import var


def example():

    var_coef = pd.read_csv("varcoef.csv").values.T
    var_coef = var_coef.flatten().reshape(-1, 3, 3)
    var_coef = np.swapaxes(var_coef, 1, 2)

    sigma2 = pd.read_csv("observation_noise_covariance.csv")
    sigma2 = sigma2.values

    var_res = var.VARAnalyzer(var_coef, sigma2)
    var_res.compute_cross_spectra()

    freqs = var_res.freqs
    amp_spectra = var_res.amplitude_spectra
    phase_spectra = var_res.phase_spectra
    power_spectra = var_res.power_spectra

    num_ts = 3
    fig, axes = plt.subplots(num_ts, num_ts, sharex=True)
    for i in range(num_ts):
        for j in range(num_ts):
            # axes[i, j].set_xscale("log")
            axes[i, j].set_xlim(0.0, 0.5)
            if i > j:
                axes[i, j].plot(freqs, phase_spectra[:, i, j])
                axes[i, j].set_ylim(-4, 4)
            elif i == j:
                axes[i, j].plot(freqs, power_spectra[:, i])
                axes[i, j].set_ylim(0.1, 100)
                axes[i, j].set_yscale("log")
            else:
                axes[i, j].plot(freqs, amp_spectra[:, i, j])
                axes[i, j].set_ylim(1e-4, 1e2)
                axes[i, j].set_yscale("log")
    plt.tight_layout()
    plt.show()
    plt.close()

    coh = var_res.coherency
    fig, axes = plt.subplots(num_ts, num_ts, sharex=True)
    for i in range(num_ts):
        for j in range(num_ts):
            axes[i, j].set_xlim(0.0, 0.5)
            if i < j:
                axes[i, j].plot(freqs, coh[:, i, j])
                if i != j:
                    axes[i, j].set_ylim(0.00, 1.00)
            elif i == j:
                axes[i, j].plot(freqs, power_spectra[:, i])
                axes[i, j].set_ylim(0.1, 100)
                axes[i, j].set_yscale("log")
            else:
                axes[i, j].remove()
    plt.tight_layout()
    plt.show()
    plt.close()

    decomp_pspec = var_res.decomposed_powerspectra
    rel_pcontrib = var_res.relative_power_contribution
    fig, axes = plt.subplots(num_ts, 2, sharex=True)
    for i in range(num_ts):
        for j in range(num_ts):
            axes[i, 0].plot(freqs, decomp_pspec[:, i, j])
            under = 0 if j == 0 else rel_pcontrib[:, i, j - 1]
            upper = rel_pcontrib[:, i, j]
            axes[i, 1].fill_between(freqs, under, upper)
        # axes[i, 0].set_xscale("log")
        # axes[i, 1].set_xscale("log")
        axes[i, 1].set_ylim(0, 1)
    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    example()


if __name__ == "__main__":
    main()
