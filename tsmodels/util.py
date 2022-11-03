"""AR model fit module.

TODO:
    *(Omma) Chnage the argument name 'num' for the number of element.
     This seems not good name.
"""
import numpy as np


def compute_autocovariance_matrix(a: np.ndarray, num: int) -> np.ndarray:
    """Compute auto-covariance matrix.
    """
    auto_cov_matrix = compute_covariance_matrix_base(a, a, num)
    return auto_cov_matrix


def compute_covariance_funciton_base(a1: np.ndarray, a2: np.ndarray):
    covs = np.correlate(a1 - a1.mean(), a2 - a2.mean(), "full")
    covs = covs[a1.size-1:]
    return covs


def compute_covariance_matrix_base(a1: np.ndarray, a2: np.ndarray,
                                   num: int) -> np.ndarray:
    """Compute base covariance matrix.
    """
    if a1.size != a2.size:
        raise ValueError(
            f"a and v must be the same size. a1 has {len(a1)}, "
            f"but a2 has {len(a2)}."
        )
    num_series = a1.size
    covs = compute_covariance_funciton_base(a1, a2) / num_series
    ids = np.arange(num)
    ids_cov_matirx = np.abs(ids - ids[:, None])
    cov_matrix = covs[ids_cov_matirx]
    return cov_matrix


def compute_crosscovariance_matrix(a1: np.ndarray, a2: np.ndarray,
                                   num: int) -> np.ndarray:
    """Compute cross-covariance matrix.
    """
    cross_cov_matirx = compute_covariance_matrix_base(a1, a2, num)
    return cross_cov_matirx
