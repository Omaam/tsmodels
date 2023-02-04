"""State space model module.
"""
from numpy.typing import ArrayLike
import numpy as np


class VARStateSpaceModel:
    """State space model of Vector autoregression (VAR) model.

    Attributes:
        var_coef (np.ndarray): VAR coefficients, whose shape must be
                (var_order, num_series, num_series).
        var_order (int): VAR order.
        num_series (int): Number of series.
        num_states (int): Number of states, which is determined with
            'var_order * num_seires'.
        transition_matrix: Transition matrix.
    """

    def __init__(self, var_coef: ArrayLike, noise_cov=None):
        """Initialization.

        Args:
            var_coef (ArrayLike): VAR coefficients, whose shape must be
                (var_order, num_series, num_series).
            noise_cov (ArrayLike, optional): Covariance of noises.
        """

        self.var_coef = var_coef

        self.var_order = var_coef.shape[0]
        self.num_series = var_coef.shape[1]
        self.num_states = self.var_order * self.num_series

        self.transition_matrix = self._get_transition_matrix()

        if noise_cov is None:
            # Generate positive semidefinite matrix by multiplying.
            noise_cov = np.random.randn(self.num_series, self.num_series)
            noise_cov = noise_cov @ noise_cov.T
        self.noise_cov = noise_cov

    def sample(self, num_iter: int, init_state: ArrayLike = None):
        """Sample series.

        Args:
            num_iter (int): Number of iterations.
            init_state (ArrayLike, optional): Initial state.
        """

        state = np.random.randn(self.num_states)[:, None] \
            if init_state is None else init_state[:, None]

        states = np.zeros((num_iter, self.num_states))
        for i in range(num_iter):
            state = self.transition_matrix @ state

            noise_state = np.random.multivariate_normal(
                np.zeros(self.num_series), self.noise_cov)
            noise_state = np.pad(noise_state,
                                 [0, self.num_states-self.num_series])
            state += noise_state[:, None]

            states[i] = np.squeeze(state)
        return states

    def _get_transition_matrix(self):
        if self.var_coef.ndim != 3:
            raise ValueError("'var_coef' must be 'ndim == 3'")

        var_coef_block = np.hstack(self.var_coef)
        past_block = np.eye((self.var_order - 1) * self.num_series)
        past_block = np.pad(past_block, [[0, 0], [0, self.num_series]])
        transition_matrix = np.vstack([var_coef_block, past_block])

        return transition_matrix
