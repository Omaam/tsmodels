"""VAR tests.
"""
import unittest

import numpy as np


from tsmodels.result.var import VarAnalyzer


class VARTest(unittest.TestCase):
    def testImpulseResponseShape(self):
        """
        """
        num_lagsteps = 20

        # Test for VAR(1).
        tester_order1 = VarAnalyzer(*self._get_testcase_order_1())
        irf_order1 = tester_order1.compute_impulse_response(
            num_lagsteps, orthogonal=True)
        expected_order1 = (20, 2, 2)
        self.assertEqual(expected_order1, irf_order1.shape)

        # Test for VAR(2).
        tester_order2 = VarAnalyzer(*self._get_testcase_order_2())
        irf_order2 = tester_order2.compute_impulse_response(num_lagsteps)
        expected_order2 = (20, 2, 2)
        self.assertEqual(expected_order2, irf_order2.shape)

    def testImpulseResponseFirstDiag(self):
        """
        """
        num_lagsteps = 20

        # Test for VAR(1) with `orthogonal=False`.
        tester_order1 = VarAnalyzer(*self._get_testcase_order_1())
        irf_order1 = tester_order1.compute_impulse_response(
            num_lagsteps, orthogonal=False)
        expected = [1., 1.]
        np.testing.assert_allclose(expected, np.diag(irf_order1[0]))

        # Test for VAR(1) with `orthogonal=True`.
        tester_order1 = VarAnalyzer(*self._get_testcase_order_1())
        irf_order1 = tester_order1.compute_impulse_response(
            num_lagsteps, orthogonal=True)
        expected = [1., 1.]
        np.testing.assert_allclose(expected, np.diag(irf_order1[0]))

        # Test for VAR(2) with `orthogonal=False`.
        tester_order2 = VarAnalyzer(*self._get_testcase_order_2())
        irf_order2 = tester_order2.compute_impulse_response(
            num_lagsteps, orthogonal=False)
        expected = [1., 1.]
        np.testing.assert_allclose(expected, np.diag(irf_order2[0]))

        # Test for VAR(2) with `orthogonal=True`.
        tester_order2 = VarAnalyzer(*self._get_testcase_order_2())
        irf_order2 = tester_order2.compute_impulse_response(
            num_lagsteps, orthogonal=True)
        expected = [1., 1.]
        np.testing.assert_allclose(expected, np.diag(irf_order2[0]))

    def _get_testcase_order_1(self):
        coefs = np.array([
            [[0.6, 0.3],
             [0.1, 0.8]],
        ])
        noise_cov = np.array(
            [[4.0, 1.2],
             [1.2, 1.0]]
        )
        return coefs, noise_cov

    def _get_testcase_order_2(self):
        coefs = np.array([
            [[0.2, 0.3],
             [0.1, 0.1]],
            [[0.1, 0.2],
             [0.2, 0.1]],
        ])
        noise_cov = np.array(
            [[4.0, 1.2],
             [1.2, 1.0]]
        )
        return coefs, noise_cov

    def _get_testcase_order_10(self):
        coefs = 0.1 * np.random.randn(10, 2, 2)
        noise_cov = np.array(
            [[4.0, 1.2],
             [1.2, 1.0]]
        )
        return coefs, noise_cov


if __name__ == "__main__":
    unittest.main()
