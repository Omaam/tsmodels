"""Example main.
"""
import numpy as np

from tsmodels.result import var
import dataset


def main():
    arcoef = dataset.get_arcoef()
    W = dataset.get_noise_covariance()
    var_analyzer = var.VarAnalyzer(arcoef, W)
    var_analyzer.compute_crossspectra()
    freqs = var_analyzer._generate_frequency()
    A = var_analyzer._compute_arcoef_fourier(freqs)

    b0 = np.linalg.inv(A[0])
    b0_H = np.conjugate(b0.T)

    p0 = b0 @ W @ b0_H
    print(p0)


if __name__ == "__main__":
    main()
