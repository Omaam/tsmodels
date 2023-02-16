"""VAR analysis.
"""
import numpy as np


class VarAnalyzer:
    def __init__(self, arcoef, W):
        self.arcoef = np.array(arcoef)
        self.arorder = arcoef.shape[0]
        self.num_series = arcoef.shape[1]

        self.W = W
        self.P = None

    def compute_crossspectra(self, num_freqs=None):
        freqs = self._generate_frequency(num_freqs)
        A = self._compute_arcoef_fourier(freqs)
        B = np.linalg.inv(A)
        B_H = np.conjugate(np.transpose(B, axes=[0, 2, 1]))
        W = self.W

        P = B @ W @ B_H

        self.num_freqs = num_freqs
        self.freqs = freqs = freqs
        self.P = P

        return P

    def _check_computation_crossspectra(self):
        if self.P is None:
            raise AttributeError("you must do 'compute_crossspectra'")

    def _compute_arcoef_fourier(self, freqs):
        a0 = np.diag(np.repeat(-1, self.num_series))
        arcoef = np.insert(self.arcoef, 0, a0, axis=0)

        phases = -2j * np.pi * freqs[:, None] * np.arange(self.arorder + 1)
        A = arcoef * np.exp(phases)[:, :, None, None]
        A = A.sum(axis=1)

        self.A = A
        return A

    def _generate_frequency(self, num_freqs=None):
        num_freqs = 201 if num_freqs is None else num_freqs
        freq_edges = np.linspace(0, 0.5, num_freqs, endpoint=True)
        freqs = (freq_edges[1:] + freq_edges[:-1]) / 2
        return freqs

    @property
    def amplitude_spectra(self):
        self._check_computation_crossspectra()
        return np.abs(self.P)

    @property
    def coherency(self):
        self._check_computation_crossspectra()

        alpha_jk = self.amplitude_spectra**2
        p_jj = np.real(np.diagonal(self.cross_spectra, axis1=1, axis2=2))
        p_kk = np.real(np.diagonal(self.cross_spectra, axis1=1, axis2=2))
        coherency = alpha_jk / p_jj[:, :, None] / p_kk[:, None, :]

        return coherency

    @property
    def cross_spectra(self):
        self._check_computation_crossspectra()
        return self.P

    @property
    def decomposed_powerspectra(self):
        self._check_computation_crossspectra()

        W = self.W
        B = np.linalg.inv(self.A)

        decomp_pspec = np.abs(B)**2 * np.diag(np.abs(W))
        decomp_pspec = np.cumsum(decomp_pspec, axis=2)
        return decomp_pspec

    @property
    def frequency(self):
        self._check_computation_crossspectra()
        return self.freqs

    @property
    def relative_power_contribution(self):
        self._check_computation_crossspectra()
        decomp_pspec = self.decomposed_powerspectra
        rel_pcontrib = decomp_pspec / decomp_pspec[:, :, -1][:, :, None]
        return rel_pcontrib

    @property
    def phase_spectra(self):
        self._check_computation_crossspectra()
        return np.angle(self.cross_spectra)

    @property
    def power_spectra(self):
        self._check_computation_crossspectra()
        power_spectra = np.diagonal(self.cross_spectra, axis1=1, axis2=2)
        return power_spectra
