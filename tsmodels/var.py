"""VAR analysis.
"""
import matplotlib.pyplot as plt
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

        ms = np.arange(self.arorder+1)
        phases = -2j*np.pi*freqs[:, None, None, None] * ms[:, None, None]
        arcoef_fourier = arcoef_a0 * np.exp(phases)

        arcoef_fourier = arcoef_fourier.sum(axis=1)

        self.arcoef_fourier = arcoef_fourier
        self.arcoef_fourier_inv = np.linalg.inv(arcoef_fourier)

    def compute_cross_spectrum(self, freqs=None):
        if freqs is None:
            freq_edges = np.linspace(0, 0.5, 101, endpoint=True)
            freqs = (freq_edges[1:] + freq_edges[:-1]) / 2
        self.freqs = freqs

        self.compute_arcoef_fourier(freqs)

        arcoef_fourier_inv_conjugate = np.empty(
            (len(freqs), *self.arcoef.shape[1:]),
            dtype=np.complex128)
        for i in range(len(freqs)):
            arcoef_fourier_inv_conjugate[i] = np.matrix(
                self.arcoef_fourier_inv[i]).getH()
        cross_spectrum = self.arcoef_fourier_inv @ self.obs_noise_matrix \
            @ arcoef_fourier_inv_conjugate
        self.cross_spectrum = cross_spectrum
        return cross_spectrum

    @property
    def amplitude_spectrum(self):
        if self.cross_spectrum is None:
            raise AttributeError(
                "you must do 'compute_cross_spectrum' before "
                "getting amplitude spectrum.")
        return np.abs(self.cross_spectrum)

    @property
    def coherency(self):
        """Coherency.

        TODO:
            * This DON'T work well. Must be repaired.
        """
        if self.cross_spectrum is None:
            raise AttributeError(
                "you must do 'compute_cross_spectrum' before "
                "getting coherency.")

        js = np.arange(3)
        ks = np.arange(3)

        # print(np.real(self.cross_spectrum[:, js[:, None], js[:, None]])[0])
        # print(np.real(self.cross_spectrum[:, ks[None, :], ks[None, :]])[0])
        upper = self.amplitude_spectrum**2
        under = np.real(self.cross_spectrum[:, ks[None, :], ks[None, :]]) \
            * np.real(self.cross_spectrum[:, js[:, None], js[:, None]])
        # print(upper[0])
        # print(under[0])
        # print(coherency)
        coherency = upper / under
        return coherency

    @property
    def frequency(self):
        if self.cross_spectrum is None:
            raise AttributeError(
                "you must do 'compute_cross_spectrum' before "
                "getting frequency.")
        return self.freqs

    @property
    def phase_spectrum(self):
        if self.cross_spectrum is None:
            raise AttributeError(
                "you must do 'compute_cross_spectrum' before "
                "getting phase spectrum.")
        return np.angle(self.cross_spectrum)


def main():
    np.random.seed(0)
    arcoef = np.random.normal(0, 1, (8, 3, 3))
    obs_noise_matrix = np.diag(np.random.normal(0, 1, 3))
    var = VARAnalyzer(arcoef, obs_noise_matrix)
    var.compute_cross_spectrum()
    freqs = var.freqs
    amp_specs = var.amplitude_spectrum
    phase_specs = var.phase_spectrum

    fig, axes = plt.subplots(3, 3, sharex=True)
    for i in range(3):
        for j in range(3):
            if i >= j:
                axes[i, j].plot(freqs, amp_specs[:, i, j])
            else:
                axes[i, j].plot(freqs, phase_specs[:, i, j])
    plt.tight_layout()
    # plt.show()
    plt.close()

    coherency = var.coherency
    fig, axes = plt.subplots(3, 3, sharex=True)
    for i in range(3):
        for j in range(3):
            if i >= j:
                axes[i, j].plot(freqs, coherency[:, i, j])
            else:
                axes[i, j].remove()
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
