"""Example for VARStateSpaceModel.
"""
import matplotlib.pyplot as plt
import numpy as np

import tsmodels


def main():

    np.random.seed(0)

    var_order = 3
    num_series = 2
    var_coef = 0.1 * np.random.randn(var_order, num_series, num_series)

    num_burnin = 64
    num_results = 128

    ssmvar = tsmodels.VARStateSpaceModel(var_coef)
    print(ssmvar.transition_matrix)

    states_burnin = ssmvar.sample(num_burnin, np.zeros(var_order*num_series))
    print("Finish burn-in!")
    states_results = ssmvar.sample(num_results, states_burnin[-1])
    print("Finish sampling!")

    width_ratios = [num_burnin//num_burnin, num_results//num_burnin]
    fig, ax = plt.subplots(num_series, 2, sharex="col", sharey="row",
                           figsize=(7, 2*num_series),
                           width_ratios=width_ratios)
    for i in range(num_series):
        ax[i, 0].plot(np.arange(states_burnin.shape[0]), states_burnin[:, i])
        ax[i, 1].plot(np.arange(states_results.shape[0]), states_results[:, i])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
