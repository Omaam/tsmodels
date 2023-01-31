"""VAR analysis.
"""
import numpy as np


class VARAnalyzer:
    def __init__(self, arcoef, sigma2):
        self.arcoef = np.array(arcoef)
        self.arorder = arcoef.shape[0]
        self.num_ts = arcoef.shape[1]

        self.W = np.diag(sigma2)
        self.P = None

    def compute_cross_spectrum(self, freqs=None):
        if freqs is None:
            freq_edges = np.linspace(0, 0.5, 101, endpoint=True)
            freqs = (freq_edges[1:] + freq_edges[:-1]) / 2
        self.freqs = freqs = np.asarray(freqs)

        A = self._compute_arcoef_fourier(freqs)
        B = np.linalg.inv(A)
        arcoef_fourier_inv_conjugate = np.conjugate(np.transpose(
            B, axes=[0, 2, 1]))

        P = B
        P = B @ self.W
        P = P @ arcoef_fourier_inv_conjugate
        self.P = P

        return P

    def _check_computation_crossspectrum(self):
        if self.P is None:
            raise AttributeError("you must do 'compute_cross_spectrum'")

    def _compute_arcoef_fourier(self, freqs):
        a0 = np.diag(np.repeat(-1, self.num_ts))
        arcoef_a0 = np.insert(self.arcoef, 0, a0, axis=0)

        phases = -2j * np.pi * freqs[:, None] * np.arange(self.arorder + 1)
        A = arcoef_a0 * np.exp(phases)[:, :, None, None]
        A = A.sum(axis=1)

        self.A = A
        return A

    @property
    def amplitude_spectrum(self):
        self._check_computation_crossspectrum()
        return np.abs(self.P)

    @property
    def coherency(self):
        self._check_computation_crossspectrum()

        alpha_jk = self.amplitude_spectrum**2
        p_jj = np.real(np.diagonal(self.P, axis1=1, axis2=2))
        p_kk = np.real(np.diagonal(self.P, axis1=1, axis2=2))
        coherency = alpha_jk / p_jj[:, :, None] / p_kk[:, None, :]

        return coherency

    @property
    def decomposed_powerspectrum(self):
        self._check_computation_crossspectrum()

        W = self.W
        B = np.linalg.inv(self.A)
        decomp_pspec = np.abs(B)**2 * np.diag(W)**2
        return decomp_pspec

    @property
    def frequency(self):
        self._check_computation_crossspectrum()
        return self.freqs

    @property
    def relative_power_contribution(self):
        self._check_computation_crossspectrum()
        decomp_pspec = self.decomposed_powerspectrum
        rel_pcontrib = np.cumsum(decomp_pspec, axis=1) / np.sum(
                decomp_pspec, axis=1, keepdims=True)
        return rel_pcontrib

    @property
    def phase_spectrum(self):
        self._check_computation_crossspectrum()
        return np.angle(self.P)
