"""AR model fit module.
"""
import unittest
import numpy as np

import util


class TestUtil(unittest.TestCase):
    def test_compute_autocovariance_matrix(self):
        self._create_testcase_00()
        actual = util.compute_autocovariance_matrix(
            self.x, self.num_cov_element)
        expect = [[2.0/3,  0.0/3, -1.0/3],
                  [0.0/3,  2.0/3, 0.0/3],
                  [-1.0/3, 0.0/3, 2.0/3], ]
        self.assertTrue(np.array_equal(actual, expect))

    def test_compute_covariance_function_base(self):
        pass

    def test_compute_covariance_matrix_base(self):
        pass

    def test_compute_crosscovariance_matrix(self):
        self._create_testcase_00()
        actual = util.compute_crosscovariance_matrix(
            self.x, self.y, self.num_cov_element)
        expect = [[0.5/3,  0.5/3,  -0.5/3],
                  [0.5/3,  0.5/3,  0.5/3],
                  [-0.5/3, 0.5/3,  0.5/3], ]
        self.assertTrue(np.array_equal(actual, expect))

    def _create_testcase_00(self):
        self.num_cov_element = 3
        self.x = np.array([1, 2, 3])
        self.y = np.array([0, 1, 0.5])


if __name__ == "__main__":
    unittest.main()
