"""Vector Autoregressive State Space Model Tests.
"""
import unittest

import numpy as np

from tsmodels.statespacemodel.vector_autoregressive \
    import VectorAutoregressiveStateSpaceModel


class VectorAutoregressiveStateSpaceModelTest(unittest.TestCase):

    def testTransitionMatrix(self):
        coefficients = np.array([
            [[1, 2],
             [3, 4]],
            [[5, 6],
             [7, 8]]
        ])
        var_ssm = VectorAutoregressiveStateSpaceModel(
            coefficients, 0.1)

        expected = np.array(
            [[1, 2, 5, 6],
             [3, 4, 7, 8],
             [1, 0, 0, 0],
             [0, 1, 0, 0]]
        )
        actual = var_ssm.transition_matrix
        is_equal = np.array_equal(expected, actual)
        self.assertTrue(is_equal)

    def testSampleShape(self):
        coefficients = np.array([
            [[0.1, 0.0],
             [0.0, 0.1]],
            [[0.1, 0.7],
             [0.0, 0.1]]
        ])
        var_ssm = VectorAutoregressiveStateSpaceModel(
            coefficients, 0.1)
        latents, observations = var_ssm.sample(100)
        self.assertEqual((100, 4), latents.shape)
        self.assertEqual((100, 2), observations.shape)


if __name__ == "__main__":
    unittest.main()
