"""Example dataset for MAXI J1820+070.

Note:
    *Number of light curves are 3 (0.5 to 2.0, 2.0 to 5.0, and
     5.0 to 10.0 keV). The analysis tool is `statsmodels.tsa.VAR`.
     The employed VAR order is 16, which is determined by AIC.
"""
import numpy as np


def get_noise_covariance():
    cov = np.load("./rawdata/var_noise_covariance.npy")
    return cov


def get_arcoef():
    coefficients = np.load("./rawdata/var_coefficients.npy")
    return coefficients
