"""Dynamic Factor State Space Model Tests.
"""
import unittest

import numpy as np

from tsmodels.statespacemodel.dynamic_factor \
    import DynamicFactorStateSpaceModel


class DynamicFactorStateSpaceModelTest(unittest.TestCase):

    def testTransitionMatrix(self):
        coefficients = np.array([
            [[1, 2],
             [3, 4]],
            [[5, 6],
             [7, 8]]
        ])
        observation_matrix = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
        ])
        df_ssm = DynamicFactorStateSpaceModel(
            coefficients, 0.1, observation_matrix, 0.1)

        expected = np.array(
            [[1, 2, 5, 6],
             [3, 4, 7, 8],
             [1, 0, 0, 0],
             [0, 1, 0, 0]]
        )
        actual = df_ssm.transition_matrix
        is_equal = np.array_equal(expected, actual)
        self.assertTrue(is_equal)

    def testObservationMatrix(self):
        coefficients = np.array([
            [[1, 2],
             [3, 4]],
            [[5, 6],
             [7, 8]]
        ])
        observation_matrix = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
        ])
        df_ssm = DynamicFactorStateSpaceModel(
            coefficients, 0.1, observation_matrix, 0.1)

        expected = np.array(
            [[1, 2, 0, 0],
             [3, 4, 0, 0],
             [5, 6, 0, 0]]
        )
        actual = df_ssm.observation_matrix
        is_equal = np.array_equal(expected, actual)
        self.assertTrue(is_equal)

    def testSampleShape(self):
        coefficients = np.array([
            [[0.1, 0.0],
             [0.0, 0.1]],
            [[0.1, 0.7],
             [0.0, 0.1]]
        ])
        observation_matrix = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
        ])
        df_ssm = DynamicFactorStateSpaceModel(
            coefficients, 0.1, observation_matrix, 0.1)
        latents, observations = df_ssm.sample(100)
        self.assertEqual((100, 4), latents.shape)
        self.assertEqual((100, 3), observations.shape)


if __name__ == "__main__":
    unittest.main()
