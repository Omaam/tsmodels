"""VAR of State space form.
"""
from numpy.typing import ArrayLike
import numpy as np

from tsmodels.statespacemodel import core


def convert_companion_matrix(coefficients: ArrayLike):
    if coefficients.ndim != 3:
        raise ValueError("'coefficients' must be 'ndim == 3'")

    var_order, num_series, _ = coefficients.shape

    coefficients_block = np.hstack(coefficients)
    past_block = np.eye((var_order - 1) * num_series)
    past_block = np.pad(past_block, [[0, 0], [0, num_series]])
    companion_matrix = np.vstack([coefficients_block, past_block])
    return companion_matrix


def check_stable(coefficients: ArrayLike):
    companion_matrix = convert_companion_matrix(coefficients)
    eigenvalues = np.linalg.eigvals(companion_matrix)
    return (np.abs(eigenvalues) <= 1).all()


class VARStateSpaceModel(core.StateSpaceModel):
    """State space model of Vector autoregression (VAR) model.

    Attributes:
        coefficients (np.ndarray): VAR coefficients, whose shape must be
                (var_order, num_series, num_series).
        var_order (int): VAR order.
        num_latents (int): Number of latents states, which means pure
            number of observations.
    """

    def __init__(self,
                 coefficients: ArrayLike,
                 state_noise_cov: ArrayLike,
                 observation_matrix: ArrayLike,
                 observation_noise_cov: ArrayLike):
        """Initialization.

        Args:
            coefficients: coefficients.
            state_noise_cov: state noise covariance.
        """
        var_order = coefficients.shape[0]
        num_observations, num_latents = observation_matrix.shape
        num_states = var_order * num_latents

        transition_matrix = convert_companion_matrix(coefficients)

        transition_noise_cov = np.pad(state_noise_cov,
                                      [0, num_states - num_latents])

        observation_matrix = np.pad(observation_matrix,
                                    [[0, 0],
                                     [0, num_states-num_latents]])

        super().__init__(
            num_states,
            num_observations,
            transition_matrix,
            transition_noise_cov,
            observation_matrix,
            observation_noise_cov
        )

        self.coefficients = coefficients
        self.var_order = var_order
        self.num_latents = num_latents
