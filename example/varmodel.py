"""VAR model example.
"""
from statsmodels.tsa.vector_ar.var_model import VARProcess
import matplotlib.pyplot as plt
import numpy as np


from tsmodels import VARAnalyzer


def example():

    np.random.seed(0)
    arcoef = np.random.normal(0, 1, (8, 3, 3))
    obs_noise_matrix = np.diag(np.random.normal(0, 0.1, 3))
    arcoef = np.array(
        [[[0.4, 0.4],
          [0.1, 0.2]]]
    )
    obs_noise_matrix = np.diag([0.1, 0.1])

    varprocess = VARProcess(arcoef, None, obs_noise_matrix)
    ts_sim = varprocess.simulate_var(100)
    fig, ax = plt.subplots(2, sharex="col", figsize=(8, 5))
    ax[0].plot(ts_sim[:, 0])
    ax[1].plot(ts_sim[:, 1])
    plt.tight_layout()
    plt.show()
    plt.close()

    var = VARAnalyzer(arcoef, obs_noise_matrix)
    var.compute_cross_spectrum()

    freqs = var.freqs
    amp_specs = var.amplitude_spectrum
    phase_specs = var.phase_spectrum

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
    plt.show()
    plt.close()

    coh = var.coherency
    fig, axes = plt.subplots(num_ts, num_ts, sharex=True)
    for i in range(num_ts):
        for j in range(num_ts):
            axes[i, j].set_xscale("log")
            if i >= j:
                axes[i, j].plot(freqs, coh[:, i, j])
            else:
                axes[i, j].remove()
    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    example()


if __name__ == "__main__":
    main()
