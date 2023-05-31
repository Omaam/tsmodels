"""Generage synthetic data.
"""
import matplotlib.pyplot as plt
import numpy as np

from tsmodels import statespacemodel as ssm


def generate_samples():
    coefficients = np.array([
        [[0.05, -0.01],
         [0.04, 0.02]],
        [[0.06, 0.02],
         [0.01, 0.09]]
      ])
    transition_noise_scale = 1.0
    observation_matrix = np.array([
        [1.0, 0.0],
        [0.4, 1.0],
        [0.3, 0.2]
    ])
    observation_noise_scale = 0.1

    df_ssm = ssm.DynamicFactorStateSpaceModel(
        coefficients, transition_noise_scale,
        observation_matrix, observation_noise_scale)
    latents, observations = df_ssm.sample(100, seed=0)

    latents = latents[:, :2]

    np.savez(
        "data/synthetic_sample_dynamic_factor_model",
        latents=latents,
        observations=observations,
        coefficients=coefficients,
        transition_noise_scale=transition_noise_scale,
        observation_matrix=observation_matrix,
        observation_noise_scale=observation_noise_scale
    )

    return latents, observations


def plot_samples(latents, num_latents,
                 observations, num_observations):
    fig, ax = plt.subplots(num_observations, 2, sharex="col")
    for i in range(num_latents):
        ax[i, 0].plot(latents[:, i])
    for j in range(num_observations):
        ax[j, 1].plot(observations[:, j])
    plt.show()
    plt.close()


def main():

    latents, observations = generate_samples()

    synthetic_sample = np.load(
        "data/synthetic_sample_dynamic_factor_model.npz")
    latents = synthetic_sample["latents"]
    observations = synthetic_sample["observations"]
    print(observations.shape)
    plot_samples(latents, 2, observations, 3)


if __name__ == "__main__":
    main()
