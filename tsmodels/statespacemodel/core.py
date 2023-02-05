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
                 num_states: int,
                 num_observations: int,
                 transition_matrix: ArrayLike,
                 transition_noise_cov: ArrayLike,
                 observation_matrix: ArrayLike,
                 observation_noise_cov: ArrayLike):
        """Initialization.

        Args:
            num_states: Number of states.
            num_observations: Number of observations.
            transition_matrix: transition matrix.
            transition_noise_cov: transition noise covariance.
            observation_matrix: observation matrix.
            observation_noise_cov: observation noise covariance.
        """
        self.num_observations = num_observations
        self.num_states = num_states
        self.observation_matrix = observation_matrix
        self.observation_noise_cov = observation_noise_cov
        self.transition_matrix = transition_matrix
        self.transition_noise_cov = transition_noise_cov

    def sample(self, num_timesteps: int, init_state: ArrayLike = None):
        """Sample series.

        Args:
            num_timesteps (int): Number of iterations.
            init_state (ArrayLike, optional): Initial state.
        """

        state = np.zeros(self.num_states)[:, None] \
            if init_state is None else init_state[:, None]

        states = np.zeros((num_timesteps, self.num_states))
        observations = np.zeros((num_timesteps, self.num_observations))
        for i in range(num_timesteps):

            state = self.transition_matrix @ state
            noise = np.random.multivariate_normal(
                np.zeros(self.num_states),
                self.transition_noise_cov)[:, None]

            states[i] = np.squeeze(state + noise)
            observations[i] = np.squeeze(
                self.observation_matrix @ states[i])

        return states, observations
