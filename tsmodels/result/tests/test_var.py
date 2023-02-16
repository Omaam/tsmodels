"""Unit test for var module.
"""
import unittest

import numpy as np

from tsmodels.result import var


def generate_random_arcoef(num_series, arorder):
    arcoef = np.random.randn(arorder, num_series, num_series)
    W = np.random.randn(num_series, num_series)
    return arcoef, W


def compute_arcoef_fourier(arcoef, freqs):
    """Compute AR coefficients by iteration.
    """
    arorder, num_series, _ = arcoef.shape

    a0 = np.diag(np.repeat(-1, num_series))
    arcoef = np.insert(arcoef, 0, a0, axis=0)
    A = np.zeros((freqs.size, num_series, num_series),
                 dtype=np.complex64)
    for i, f in enumerate(freqs):
        for m in range(arorder+1):
            for j in range(num_series):
                for k in range(num_series):
                    A[i, j, k] += arcoef[m, j, k] * np.exp(
                        -2j * np.pi * m * f)
    return A


def compute_crossspectra(A, W):
    P = np.zeros(A.shape, dtype=np.complex64)
    for i in range(A.shape[0]):
        B = np.linalg.inv(A[i])
        B_H = np.conjugate(B.T)
        P[i] = B @ W @ B_H
    return P


class YourClassTest(unittest.TestCase):
    def setUp(self):

        num_series = 2
        arorder = 3

        arcoef, W = generate_random_arcoef(num_series, arorder)
        var_analyzer = var.VarAnalyzer(arcoef, W)

        self.arcoef = arcoef
        self.arorder = arorder
        self.num_series = num_series
        self.var_analyzer = var_analyzer
        self.W = W

    def test_compute_arcoef_fourier(self):
        freqs = self.var_analyzer._generate_frequency()
        A_expect = self.var_analyzer._compute_arcoef_fourier(freqs)
        A_actual = compute_arcoef_fourier(self.arcoef, freqs)
        self.assertTrue(np.allclose(A_actual, A_expect))

    def test_compute_crossspectrum(self):
        freqs = self.var_analyzer._generate_frequency()
        A_actual = compute_arcoef_fourier(self.arcoef, freqs)

        cs_actual = compute_crossspectra(A_actual, self.W)
        cs_expect = self.var_analyzer.compute_crossspectra()
        self.assertTrue(np.allclose(cs_actual, cs_expect))

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
