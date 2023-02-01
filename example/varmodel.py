"""VAR model example.
"""
from statsmodels.tsa.vector_ar.var_model import VARProcess
import matplotlib.pyplot as plt
import numpy as np


from tsmodels import VARAnalyzer


def example():

    arcoef = np.array(
        [[[0.4, 0.4],
          [0.1, 0.4]]]
    )
    sigma2 = [0.1, 0.1]

    varprocess = VARProcess(arcoef, None, np.diag(sigma2))
    ts_sim = varprocess.simulate_var(100)
    fig, ax = plt.subplots(2, sharex="col", figsize=(8, 5))
    ax[0].plot(ts_sim[:, 0])
    ax[1].plot(ts_sim[:, 1])
    plt.tight_layout()
    # plt.show()
    plt.close()

    var = VARAnalyzer(arcoef, np.diag(sigma2))
    var.compute_cross_spectra()

    freqs = var.freqs
    amp_specs = var.amplitude_spectra
    phase_specs = var.phase_spectra

    num_ts = 2
    fig, axes = plt.subplots(num_ts, num_ts, sharex=True)
    for i in range(num_ts):
        for j in range(num_ts):
            axes[i, j].set_xscale("log")
            if i >= j:
                axes[i, j].plot(freqs, amp_specs[:, i, j])
            else:
                axes[i, j].plot(freqs, phase_specs[:, i, j])
    plt.tight_layout()
    # plt.show()
    plt.close()

    coh = var.coherency
    fig, axes = plt.subplots(num_ts, num_ts, sharex=True)
    for i in range(num_ts):
        for j in range(num_ts):
            axes[i, j].set_xscale("log")
            if i <= j:
                axes[i, j].plot(freqs, coh[:, i, j])
                if i != j:
                    axes[i, j].set_ylim(-3.15, 3.15)
            else:
                axes[i, j].remove()
    plt.tight_layout()
    # plt.show()
    plt.close()

    decomp_pspec = var.decomposed_powerspectra
    rel_pcontrib = var.relative_power_contribution
    fig, axes = plt.subplots(num_ts, 2, sharex=True)
    for i in range(num_ts):
        for j in range(num_ts):
            axes[i, 0].plot(freqs, decomp_pspec[:, i, j])
            under = 0 if j == 0 else rel_pcontrib[:, i, j - 1]
            upper = rel_pcontrib[:, i, j]
            axes[i, 1].fill_between(freqs, under, upper)
        axes[i, 0].set_xscale("log")
        axes[i, 1].set_xscale("log")
        axes[i, 1].set_ylim(0, 1)
    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    example()


if __name__ == "__main__":
    main()
