"""Results.
"""
import numpy as np


def get_noise_covariance():
    cov = np.array([
        [-0.589,  0.407,  0.118],
        [0.407, -0.431,  -0.41],
        [-0.118,  0.41,   0.557]
    ])
    return cov


def get_arcoef():
    arcoef = np.array([
        0.547, 0.248, -0.105, -0.402, -0.276, -0.44, 0.652, 0.53, 0.576,
        0.011, -0.044, 0.041, 0.184, 0.47, 0.148, -0.481, -0.299, 0.284,
        0.0, 0.367, 0.373, -0.208, 0.693, 0.637, 0.113, -0.027, -0.43,
        0.125, 0.35, 0.23, 0.184, 0.819, 0.372, -0.482, -0.512, -0.567,
        -0.118, 0.047, -0.008, -0.583, 0.411, -0.095, 0.617, -0.081, -0.099,
        0.252, -0.005, -0.185, 0.835, 0.159, -0.296, -0.207, -0.561, 0.309,
        0.445, -0.542, 0.03, 0.352, -1.179, 0.091, -0.926, 0.634, 0.099,
        -0.088, -0.664, 0.147, -0.16, -0.562, 0.245, 0.303, 1.032, -0.313
    ]).reshape((8, 3, 3))
    return arcoef