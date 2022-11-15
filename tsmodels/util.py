"""AR model fit module.

TODO:
    *(Omma) Chnage the argument name 'num' for the number of element.
     This seems not good name.
"""
import numpy as np


def assert_equal_size(a1: np.ndarray, a2: np.ndarray) -> None:
    if a1.size != a2.size:
        raise ValueError(
            f"a and v must be the same size. a1 has {len(a1)}, "
            f"but a2 has {len(a2)}."
        )


def compute_autocovariance_funciton(a: np.ndarray, num: int) -> np.ndarray:
    """Compute auto-covariance function"""
    covs = compute_crosscovariance_funciton(a, a, num)
    return covs


def compute_autocovariance_matrix(a: np.ndarray, num: int) -> np.ndarray:
    """Compute auto-covariance matrix."""
    cov_matrix = compute_crosscovariance_matrix(a, a, num)
    return cov_matrix


def compute_covariance_matrix_from_covariances(
        covs: np.ndarray) -> np.ndarray:
    """Compute base covariance matrix."""
    ids = np.arange(covs.size)
    ids_cov_matirx = np.abs(ids - ids[:, None])
    cov_matrix = covs[ids_cov_matirx]
    return cov_matrix


def compute_crosscovariance_funciton(a1: np.ndarray, a2: np.ndarray,
                                     num: int) -> np.ndarray:
    """Compute cross-covariance funciton."""
    assert_equal_size(a1, a2)
    covs = np.correlate(a1 - a1.mean(), a2 - a2.mean(), "full")
    covs = covs[a1.size-1:] / a1.size
    return covs[:num]


def compute_crosscovariance_matrix(a1: np.ndarray, a2: np.ndarray,
                                   num: int) -> np.ndarray:
    """Compute cross-covariance matrix."""
    covs = compute_crosscovariance_funciton(a1, a2, num)
    cov_matrix = compute_covariance_matrix_from_covariances(covs)
    return cov_matrix


def compute_lowerorder_arcoef(arcoefs: np.ndarray) -> np.ndarray:
    """Compute AR coefficients for lower order AR models.

    Returns
        * arcoef_matrix:
            AR coefficients matrix whose shape is (m-1, m-1).
            (i, j) represents a^j_i.
    """

    arcoefs_list = []

    def estimate_arcoef_recursive(c):
        c_next = (c[:-1] + c[-1] * c[-2::-1]) / (1 - c[-1]**2)
        if len(c) == 1:
            return
        arcoefs_list.append(list(c_next))
        estimate_arcoef_recursive(c_next)

    estimate_arcoef_recursive(arcoefs)
    return arcoefs_list


def compute_parcor(arcoefs: np.ndarray) -> np.ndarray:
    """Compute PARCOR from AR model coefficients.

    Returns
        * parcor: PARCOR(partial autocorrelation coefficient)
    """
    arcoef_matrix = compute_lowerorder_arcoef(arcoefs)
    parcor = np.diag(arcoef_matrix)
    return parcor
