"""Test for model module.
"""
import math

import matplotlib.pyplot as plt
import numpy as np

import tsmodels


def plot_power_spectrum(armodel):
    freqs = np.linspace(0, 0.5, 1000)
    powers = armodel.compute_power_spectrum(freqs)
    fig, ax = plt.subplots()
    ax.plot(freqs, powers)
    ax.set_yscale("log")
    # plt.show()
    plt.close()


def plot_impulse_response_function(armodel):
    irf = armodel.compute_impulse_response_function(20)
    fig, ax = plt.subplots()
    ax.plot(irf)
    ax.set_ylim(-1, 2)
    # plt.show()
    plt.close()


def example():
    armodel = tsmodels.ARModel(ar_coefs)

    plot_power_spectrum(armodel)

    plot_impulse_response_function(armodel)


if __name__ == "__main__":
    ar_coefs = [0.9*math.sqrt(3), -0.81]
    example()
