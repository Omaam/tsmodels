"""VAR of State space form.
"""
from numpy.typing import ArrayLike
import numpy as np

from tsmodels.statespacemodel.state_spece_model import StateSpaceModel
import tsmodels.core as gen_core


def make_companion_matrix(coefficients: ArrayLike):
    if coefficients.ndim != 3:
        raise ValueError("'coefficients' must be 'ndim == 3'")

    var_order, num_series, _ = coefficients.shape

    coefficients_block = np.hstack(coefficients)
    past_block = np.eye((var_order - 1) * num_series)
    past_block = np.pad(past_block, [[0, 0], [0, num_series]])
    companion_matrix = np.vstack([coefficients_block, past_block])
    return companion_matrix


def check_stable(coefficients: ArrayLike):
    companion_matrix = make_companion_matrix(coefficients)
    eigenvalues = np.linalg.eigvals(companion_matrix)
    return (np.abs(eigenvalues) <= 1).all()


class VectorAutoregressiveStateSpaceModel(StateSpaceModel):
    """State space model of Vector autoregression (VAR) model.

    Attributes:
        coefficients (np.ndarray): VAR coefficients, whose shape must be
                (var_order, num_series, num_series).
        var_order (int): VAR order.
        num_latents (int): Number of latents states, which means pure
            number of observations.
    """

    def __init__(self,
                 coefficients,
                 level_scale,
                 observation_noise_scale=0.0
                 ):
        """Initialization.

        Args:
            coefficients: coefficients.
            state_noise_cov: state noise covariance.
        """
        coefficients = gen_core.as_array(coefficients)
        level_scale = gen_core.as_array(level_scale)
        observation_noise_scale = gen_core.as_array(observation_noise_scale)

        order, ndims, _ = coefficients.shape

        if len(level_scale) == 1:
            level_scale = np.repeat(level_scale, ndims)

        if len(observation_noise_scale) == 1:
            observation_noise_scale = np.repeat(
                observation_noise_scale, ndims)

        transition_matrix = make_companion_matrix(coefficients)

        transition_noise_cov = np.diag(
            np.concatenate([level_scale, np.zeros((order-1) * ndims)]))

        observation_matrix = np.concatenate(
            [np.diag(np.ones(ndims)),
             np.zeros((ndims, (order-1) * ndims))],
            axis=1)

        observation_noise_cov = np.diag(observation_noise_scale)

        super(VectorAutoregressiveStateSpaceModel, self).__init__(
            transition_matrix,
            transition_noise_cov,
            observation_matrix,
            observation_noise_cov
        )
