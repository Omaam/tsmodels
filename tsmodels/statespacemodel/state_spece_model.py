"""Core module for state space model.
"""
from numpy.typing import ArrayLike
import numpy as np


class StateSpaceModel():
    """Core of state space model.

    Attributes:
        num_states: Number of states.
        num_observations: Number of observations.
        observation_matrix: observation matrix.
        observation_noise_cov: observation noise covariance.
        transition_matrix: transition matrix.
        transition_noise_cov: transition noise covariance.
    """

    def __init__(self,
                 transition_matrix,
                 transition_noise_cov,
                 observation_matrix,
                 observation_noise_cov
                 ):
        """Initialization.

        Args:
            num_states: Number of states.
            num_observations: Number of observations.
            transition_matrix: transition matrix.
            transition_noise_cov: transition noise covariance.
            observation_matrix: observation matrix.
            observation_noise_cov: observation noise covariance.
        """
        observation_size, latent_size = observation_matrix.shape

        self.latent_size = latent_size
        self.transition_matrix = transition_matrix
        self.transition_noise_cov = transition_noise_cov
        self.observation_size = observation_size
        self.observation_matrix = observation_matrix
        self.observation_noise_cov = observation_noise_cov

    def sample(self, num_timesteps: int,
               initial_latent_state: ArrayLike = None,
               seed=None):
        """Sample series.

        Args:
            num_timesteps (int): Number of iterations.
            init_state (ArrayLike, optional): Initial state.
        """
        np.random.seed(seed)

        latents = np.zeros((num_timesteps, self.latent_size))
        observations = np.zeros((num_timesteps, self.observation_size))

        if initial_latent_state is None:
            initial_latent_state = np.zeros(self.latent_size)
        else:
            if initial_latent_state.shape[-1] != self.latent_size:
                raise ValueError()
            latents[0] = initial_latent_state

        for i in range(num_timesteps - 1):
            latent_mean = self.transition_matrix @ latents[i]
            latent_noise = np.random.multivariate_normal(
                np.zeros(self.latent_size),
                self.transition_noise_cov)
            latents[i+1] = latent_mean + latent_noise

            observation_mean = self.observation_matrix @ latents[i+1]
            observation_noise = np.random.multivariate_normal(
                np.zeros(self.observation_size),
                self.observation_noise_cov)
            observations[i+1] = observation_mean + observation_noise

        return latents, observations
