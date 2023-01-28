"""VAR analysis.
"""
import numpy as np


class VARAnalyzer:
    def __init__(self, arcoef, obs_noise_matrix):
        self.arcoef = np.array(arcoef)
        self.arorder = arcoef.shape[0]
        self.num_ts = arcoef.shape[1]

        self.obs_noise_matrix = obs_noise_matrix

        self.cross_spectrum = None

    def compute_arcoef_fourier(self, freqs):
        a0 = np.diag(np.repeat(-1, self.num_ts))
        arcoef_a0 = np.insert(self.arcoef, 0, a0, axis=0)

        phases = -2j * np.pi * freqs[:, None] * np.arange(self.arorder + 1)
        arcoef_fourier = arcoef_a0 * np.exp(phases)[:, :, None, None]
        arcoef_fourier = arcoef_fourier.sum(axis=1)

        self.arcoef_fourier = arcoef_fourier
        return arcoef_fourier

    def compute_cross_spectrum(self, freqs=None):
        if freqs is None:
            freq_edges = np.linspace(0, 0.5, 101, endpoint=True)
            freqs = (freq_edges[1:] + freq_edges[:-1]) / 2
        self.freqs = freqs = np.asarray(freqs)

        arcoef_fourier = self.compute_arcoef_fourier(freqs)
        arcoef_fourier_inv = np.linalg.inv(arcoef_fourier)
        arcoef_fourier_inv_conjugate = np.conjugate(np.transpose(
            arcoef_fourier_inv, axes=[0, 2, 1]))

        cross_spectrum = arcoef_fourier_inv
        cross_spectrum = arcoef_fourier_inv @ self.obs_noise_matrix
        cross_spectrum = cross_spectrum @ arcoef_fourier_inv_conjugate
        self.cross_spectrum = cross_spectrum

        return cross_spectrum

    def _check_computation_crossspectrum(self):
        if self.cross_spectrum is None:
            raise AttributeError("you must do 'compute_cross_spectrum'")

    @property
    def amplitude_spectrum(self):
        self._check_computation_crossspectrum()
        return np.abs(self.cross_spectrum)

    @property
    def coherency(self):
        self._check_computation_crossspectrum()

        alpha_jk = self.amplitude_spectrum**2
        p_jj = np.real(np.diagonal(self.cross_spectrum, axis1=1, axis2=2))
        p_kk = np.real(np.diagonal(self.cross_spectrum, axis1=1, axis2=2))
        coherency = alpha_jk / p_jj[:, :, None] / p_kk[:, None, :]

        return coherency

    @property
    def frequency(self):
        self._check_computation_crossspectrum()
        return self.freqs

    @property
    def phase_spectrum(self):
        self._check_computation_crossspectrum()
        return np.angle(self.cross_spectrum)
