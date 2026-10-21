"""Muestreo, FFT y teorema de Nyquist-Shannon (apartado c)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


def magnitude_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Espectro de magnitud |X(f)| con FFT y ventana de Hann.

    La magnitud se corrige por la ganancia coherente de la ventana, de modo que
    una senoide de amplitud A produce un pico de altura A.
    """
    n = len(x)
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    mag = np.abs(X) * 2.0 / np.sum(w)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return f, mag


def cumulative_frequency(f: np.ndarray, power: np.ndarray, fraction: float,
                         fmin: float = 20.0) -> float:
    """Frecuencia por debajo de la cual se acumula `fraction` de la energía."""
    m = f >= fmin
    ff, pp = f[m], power[m]
    if ff.size == 0:
        return 0.0
    c = np.cumsum(pp)
    if c[-1] <= 0:
        return 0.0
    idx = int(np.searchsorted(c, fraction * c[-1]))
    return float(ff[min(idx, len(ff) - 1)])


def effective_bandwidth(f: np.ndarray, mag: np.ndarray, fraction: float = 0.99) -> float:
    return cumulative_frequency(f, mag ** 2, fraction)


def decimate(x: np.ndarray, fs: float, fs_target: float,
             antialias: bool) -> tuple[int, float, np.ndarray]:
    """Muestrea x(t) (ya discreta a fs) tomando 1 de cada `step` muestras.

    Sin filtro antialiasing se ve el plegado espectral; con filtro (Butterworth
    orden 8 a 0.45·fs_nuevo) se cumple la condición de Nyquist por construcción.
    Devuelve (step, fs_efectiva, x_n).
    """
    step = max(1, int(round(fs / fs_target)))
    fs_eff = fs / step
    if step == 1:
        return 1, float(fs), x.copy()
    y = x
    if antialias:
        sos = signal.butter(8, 0.45 * fs_eff, btype="low", fs=fs, output="sos")
        y = signal.sosfiltfilt(sos, x)
    return step, float(fs_eff), y[::step].copy()


@dataclass
class NyquistReport:
    fs_original: float
    fs_sampled: float
    f_max: float
    fraction: float
    step: int
    antialias: bool
    energy_above: float      # fracción de energía del original sobre el Nyquist de x(n)

    @property
    def fs_min(self) -> float:
        return 2.0 * self.f_max

    @property
    def nyquist_sampled(self) -> float:
        return self.fs_sampled / 2.0

    @property
    def ok(self) -> bool:
        return self.fs_sampled >= self.fs_min

    def as_text(self) -> str:
        l1 = (f"Original: fs = {self.fs_original:.0f} Hz → Nyquist = {self.fs_original / 2:.0f} Hz.  "
              f"Ancho de banda efectivo ({self.fraction * 100:.0f} % de la energía): "
              f"f_max = {self.f_max:.0f} Hz → Nyquist-Shannon exige fs ≥ 2·f_max = {self.fs_min:.0f} Hz.")
        verdict = "CUMPLE ✔ (fs ≥ 2·f_max)" if self.ok else "NO CUMPLE ✖ (fs < 2·f_max → aliasing)"
        l2 = (f"x(n): fs = {self.fs_sampled:.0f} Hz (decimación ×{self.step}, "
              f"Ts = {1e6 / self.fs_sampled:.1f} µs), Nyquist = {self.nyquist_sampled:.0f} Hz → {verdict}.")
        if self.antialias:
            tail = "eliminada por el filtro antialiasing (Butterworth orden 8)"
        elif self.energy_above > 1e-3:
            tail = "se pliega sobre el espectro de x(n) (aliasing)"
        else:
            tail = "despreciable, casi no hay aliasing"
        l3 = f"Energía del original sobre {self.nyquist_sampled:.0f} Hz: {self.energy_above * 100:.2f} % → {tail}."
        return "\n".join((l1, l2, l3))


def nyquist_report(f: np.ndarray, mag: np.ndarray, fs_original: float, fs_sampled: float,
                   step: int, antialias: bool, fraction: float = 0.99) -> NyquistReport:
    p = mag ** 2
    total = p.sum()
    above = float(p[f > fs_sampled / 2.0].sum() / total) if total > 0 else 0.0
    return NyquistReport(fs_original, fs_sampled, effective_bandwidth(f, mag, fraction),
                         fraction, step, antialias, above)
