"""Test for model module.
"""
import matplotlib.pyplot as plt

import tsmodels

ar_coefs = [1.55, -0.81]


def test_armodel():
    armodel = tsmodels.ARModel(ar_coefs)

    # freqs = np.linspace(0, 0.5, 1000)
    # powers = armodel.compute_power_spectrum(freqs)

    irf = armodel.compute_impulse_response_function(20)
    print(irf)

    plt.plot(irf)
    plt.ylim(-1, 2)
    plt.show()


if __name__ == "__main__":
    test_armodel()
