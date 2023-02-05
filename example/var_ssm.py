"""Example for VARStateSpaceModel.
"""
import matplotlib.pyplot as plt
import numpy as np

from tsmodels.statespacemodel import var


def define_varstatespacemodel(var_order, num_latents, num_observations):

    var_coef = search_stable_coefficients(var_order, num_latents)

    state_noise_cov = get_random_noise_covariance(3)

    observation_matrix = np.random.randn(num_observations, num_latents)
    observation_noise_cov = np.diag(np.random.randn(3))

    ssmvar = var.VARStateSpaceModel(
        var_coef,
        state_noise_cov,
        observation_matrix,
        observation_noise_cov
    )
    return ssmvar


def get_random_noise_covariance(num_series: int):
    noise_cov = np.random.randn(num_series, num_series)
    noise_cov = noise_cov @ noise_cov.T  # To be semidefinite matrix.
    return noise_cov


def search_stable_coefficients(var_order: int, num_series: int,
                               scale: float = 0.1):
    seed_counter = 0
    while 1:
        np.random.seed(seed_counter)
        coefficients = scale * np.random.randn(
            var_order, num_series, num_series)
        is_stable = var.check_stable(coefficients)
        if is_stable:
            break
        seed_counter += 1
    return coefficients


def plot_series(num_series, burnin, results, width_ratios):
    fig, ax = plt.subplots(num_series, 2, sharex="col", sharey="row",
                           figsize=(7, 2*num_series),
                           width_ratios=width_ratios)
    for i in range(num_series):
        ax[i, 0].plot(np.arange(burnin.shape[0]), burnin[:, i])
        ax[i, 1].plot(np.arange(results.shape[0]), results[:, i])
    plt.tight_layout()
    plt.show()
    plt.close()


def main():

    var_order = 3
    num_latents = 3
    num_observations = 5

    ssmvar = define_varstatespacemodel(
        var_order, num_latents, num_observations)

    num_burnin = 64
    states_burnin, observations_burnin = ssmvar.sample(num_burnin)

    num_results = 128
    states_results, observations_results = ssmvar.sample(num_results,
                                                         states_burnin[-1])

    width_ratios = [num_burnin//num_burnin, num_results//num_burnin]
    plot_series(num_latents, states_burnin, states_results, width_ratios)
    plot_series(num_observations, observations_burnin, observations_results,
                width_ratios)


if __name__ == "__main__":
    main()
