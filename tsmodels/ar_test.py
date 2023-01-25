"""Test for model module.
"""
import unittest

import numpy as np

import model


class TestARModel(unittest.TestCase):

    def test_compute_impulse_response_function(self, num_lags=50):

        ar_coef = [0.3, 0.2, 0.1]
        armodel = model.ARModel(ar_coef)

        expect = self._compute_irf_manual(ar_coef)
        actual = armodel.compute_impulse_response_function()

        self.assertTrue(np.array_equal(expect, actual))

    def _compute_irf_manual(self, ar_coef, num_lags=50) -> list:

        pad_width = num_lags - len(ar_coef)
        ar_coef = np.pad(ar_coef, (0, pad_width))

        g = []
        for i in range(num_lags + 1):
            if i == 0:
                g.append(1.0)
                continue
            g_i = 0.0
            for j in range(1, i+1, 1):
                g_i += ar_coef[j-1] * g[i-j]
            g.append(g_i)
        return g


if __name__ == '__main__':
    unittest.main()
