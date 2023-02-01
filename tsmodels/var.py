"""VAR analysis.
"""
import numpy as np


class VARAnalyzer:
    def __init__(self, arcoef, W):
        self.arcoef = np.array(arcoef)
        self.arorder = arcoef.shape[0]
        self.num_ts = arcoef.shape[1]

        self.W = W
        self.P = None

    def compute_cross_spectra(self, freqs=None):
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

    def _check_computation_crossspectra(self):
        if self.P is None:
            raise AttributeError("you must do 'compute_cross_spectra'")

    def _compute_arcoef_fourier(self, freqs):
        a0 = np.diag(np.repeat(-1, self.num_ts))
        arcoef_a0 = np.insert(self.arcoef, 0, a0, axis=0)

        phases = -2j * np.pi * freqs[:, None] * np.arange(self.arorder + 1)
        A = arcoef_a0 * np.exp(phases)[:, :, None, None]
        A = A.sum(axis=1)

        self.A = A
        return A

    @property
    def amplitude_spectra(self):
        self._check_computation_crossspectra()
        return np.abs(self.P)

    @property
    def coherency(self):
        self._check_computation_crossspectra()

        alpha_jk = self.amplitude_spectra**2
        p_jj = np.real(np.diagonal(self.P, axis1=1, axis2=2))
        p_kk = np.real(np.diagonal(self.P, axis1=1, axis2=2))
        coherency = alpha_jk / p_jj[:, :, None] / p_kk[:, None, :]

        return coherency

    @property
    def cross_spectra(self):
        return self.P

    @property
    def decomposed_powerspectra(self):
        self._check_computation_crossspectra()

        W = self.W
        B = np.linalg.inv(self.A)
        decomp_pspec = np.abs(B)**2 * np.diag(W)**2
        return decomp_pspec

    @property
    def frequency(self):
        self._check_computation_crossspectra()
        return self.freqs

    @property
    def relative_power_contribution(self):
        self._check_computation_crossspectra()
        decomp_pspec = self.decomposed_powerspectra
        rel_pcontrib = np.cumsum(decomp_pspec, axis=2)
        rel_pcontrib = rel_pcontrib / self.power_spectra[:, :, None]
        return rel_pcontrib

    @property
    def phase_spectra(self):
        self._check_computation_crossspectra()
        return np.angle(self.P)

    @property
    def power_spectra(self):
        self._check_computation_crossspectra()
        power_spectra = np.diagonal(self.cross_spectra, axis1=1, axis2=2)
        power_spectra = np.abs(power_spectra)
        return power_spectra
