"""Dynamic Factor Model.
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


class DynamicFactorStateSpaceModel(StateSpaceModel):
    """Dynamic Factor State Space Model.

    Attributes:
        coefficients (np.ndarray): VAR coefficients, whose shape must be
                (var_order, num_series, num_series).
        var_order (int): VAR order.
        num_latents (int): Number of latents states, which means pure
            number of observations.
    """

    def __init__(self,
                 coefficients,
                 transition_noise_scale,
                 observation_matrix,
                 observation_noise_scale
                 ):
        """Initialization.

        Args:
            coefficients: coefficients.
            state_noise_cov: state noise covariance.
        """
        coefficients = gen_core.as_array(coefficients)
        transition_noise_scale = gen_core.as_array(transition_noise_scale)
        observation_matrix = gen_core.as_array(observation_matrix)
        observation_noise_scale = gen_core.as_array(observation_noise_scale)

        order, ndims, _ = coefficients.shape
        observation_size = observation_matrix.shape[0]

        if ndims != observation_matrix.shape[1]:
            raise ValueError()

        if len(transition_noise_scale) == 1:
            transition_noise_scale = np.repeat(transition_noise_scale, ndims)

        if len(observation_noise_scale) == 1:
            observation_noise_scale = np.repeat(
                observation_noise_scale, observation_size)

        transition_matrix = make_companion_matrix(coefficients)

        transition_noise_cov = np.diag(
            np.concatenate([transition_noise_scale,
                            np.zeros((order-1) * ndims)])
        )

        if observation_noise_scale.shape[0] != observation_size:
            raise ValueError()

        observation_matrix = np.concatenate(
            [observation_matrix,
             np.zeros((observation_size, (order-1) * ndims))],
            axis=1
        )
        observation_noise_cov = np.diag(observation_noise_scale)

        super(DynamicFactorStateSpaceModel, self).__init__(
            transition_matrix,
            transition_noise_cov,
            observation_matrix,
            observation_noise_cov
        )
