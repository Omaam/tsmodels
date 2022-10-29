"""Time series models.
"""
from numpy.typing import ArrayLike
import numpy as np


class ARModel:
    """AR model class.
    Attribute:
        * ar_coef: ArrayLike
            AR model coefficients.
        * num_coef: int
            Number of AR model coefficients.
    """
    def __init__(self, ar_coefficients: ArrayLike):
        self.coefs = np.asarray(ar_coefficients)
        self.num_coef = self.coefs.size

    def compute_auto_covariance_function(self, num_lags=50):
        pass

    def compute_impulse_response_function(self, num_lags=50):
        irf = np.zeros(num_lags + 1)
        for i in range(num_lags + 1):
            if i == 0:
                irf[i] = 1
                continue
            g_i = np.convolve(self.coefs[:i], irf[:i], "same")
            irf[i] = g_i[i-1]
        return irf

    def compute_parcor(self, num_lags=50):
        pass

    def compute_power_spectrum(self, freqs: ArrayLike):
        js = np.arange(1, self.num_coef+1, 1)
        coef_fourier = self.coefs * np.exp(-2j * np.pi * freqs[:, None] * js)
        ar_part = np.abs(1 - np.sum(coef_fourier, axis=1))**2
        powers = 1 / ar_part
        return powers
