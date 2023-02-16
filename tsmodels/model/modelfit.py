"""AR model fit module.
"""
from numpy.typing import ArrayLike
import numpy as np

from tsmodels import util


def fit_armodel(x: ArrayLike, arorder: int):
    autocovs = util.compute_autocovariance_funciton(x, arorder + 1)
    autocov_matrix = util.compute_autocovariance_matrix(x, arorder)
    arcoef = np.linalg.solve(autocov_matrix, autocovs[1:])
    error_variance = autocovs[0] - np.sum(arcoef*autocovs[1:])
    return arcoef, error_variance
