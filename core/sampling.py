"""Muestreo, FFT y teorema de Nyquist-Shannon (apartado c)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

ANTIALIAS_ORDER = 8
ANTIALIAS_CUTOFF_RATIO = 0.45   # corte en 0.45·fs_nuevo, por debajo de su Nyquist


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


def dominant_frequency(f: np.ndarray, p: np.ndarray, fmin: float = 20.0) -> float:
    """Frecuencia del pico más alto del espectro."""
    m = f >= fmin
    return float(f[m][np.argmax(p[m])]) if m.any() else 0.0


def spectral_centroid(f: np.ndarray, p: np.ndarray, fmin: float = 20.0,
                      fmax: float = 20000.0) -> float:
    """Centroide espectral (media de frecuencias ponderada por la potencia)."""
    m = (f >= fmin) & (f <= fmax)
    tot = p[m].sum()
    return float((f[m] * p[m]).sum() / tot) if tot > 0 else 0.0


def pick_zoom_start(x: np.ndarray, fs: float, win_s: float) -> int:
    """Índice de inicio de la ventana de win_s con más energía (para los zooms)."""
    n = int(win_s * fs)
    if len(x) <= n:
        return 0
    c = np.concatenate(([0.0], np.cumsum(x.astype(np.float64) ** 2)))
    starts = np.arange(0, len(x) - n + 1, max(1, n // 2))
    return int(starts[np.argmax(c[starts + n] - c[starts])])


def decimation_step(fs: float, fs_target: float) -> int:
    """Factor de decimación entero.

    La fs efectiva es fs/step, así que casi nunca coincide con la fs pedida
    (pedir 8000 Hz sobre un archivo de 44100 Hz da 7350 Hz).
    """
    return max(1, int(round(fs / fs_target)))


def antialias_sos(fs: float, fs_new: float, order: int = ANTIALIAS_ORDER,
                  cutoff_ratio: float = ANTIALIAS_CUTOFF_RATIO):
    """Filtro paso bajo de Butterworth para el antialiasing previo a la decimación."""
    return signal.butter(order, cutoff_ratio * fs_new, btype="low", fs=fs, output="sos")


def lowpass_antialias(x: np.ndarray, fs: float, fs_new: float,
                      order: int = ANTIALIAS_ORDER,
                      cutoff_ratio: float = ANTIALIAS_CUTOFF_RATIO) -> np.ndarray:
    """Paso bajo de fase cero que prepara la señal para la decimación.

    El corte está en 0.45·fs_nueva, por debajo de su Nyquist, para que el
    contenido que se va a plegar quede atenuado.  Cuidado: un filtro paso bajo
    puede sobreimpulsar en los transitorios (hasta un 12 % en audio real), así
    que la normalización de pico debe hacerse DESPUÉS de filtrar.
    """
    return signal.sosfiltfilt(antialias_sos(fs, fs_new, order, cutoff_ratio), x)


def decimate(x: np.ndarray, fs: float, fs_target: float) -> tuple[int, float, np.ndarray]:
    """Decimación entera pura: x(n) = x(n·Ts), tomando 1 de cada `step` muestras.

    La señal debe llegar ya filtrada (ver `lowpass_antialias`) y normalizada.
    Devuelve (step, fs_efectiva, x_n).
    """
    step = decimation_step(fs, fs_target)
    return step, float(fs / step), x[::step].copy()


def aliasing_residual(f: np.ndarray, mag: np.ndarray, fs: float, fs_new: float,
                       antialias: bool) -> tuple[float, float]:
    """Energía que se pliega y energía que sobrevive al filtro antialiasing.

    Devuelve (fracción que se pliega, fracción que queda plegándose después del
    filtro), ambas como fracción de la energía total del original. La segunda
    cifra es la que permite afirmar si el aliasing quedó eliminado o solo atenuado.
    """
    p = mag ** 2
    total = float(p.sum())
    if total <= 0 or fs_new >= fs:
        return 0.0, 0.0
    m = f > fs_new / 2.0
    fold = float(p[m].sum())
    if not antialias:
        return fold / total, fold / total
    sos = antialias_sos(fs, fs_new)
    _, h = signal.sosfreqz(sos, worN=f, fs=fs)
    resid = float((p[m] * np.abs(h[m]) ** 2).sum())
    return fold / total, resid / total


@dataclass
class NyquistReport:
    fs_original: float
    fs_asked: float
    fs_sampled: float
    f_max: float
    fraction: float
    step: int
    antialias: bool
    energy_above: float      # fracción de energía del original sobre el Nyquist de x(n)
    residual: float          # fracción de esa energía que el filtro deja pasar
    cutoff_hz: float         # frecuencia de corte del filtro antialiasing

    @property
    def fs_min(self) -> float:
        return 2.0 * self.f_max

    @property
    def nyquist_sampled(self) -> float:
        return self.fs_sampled / 2.0

    @property
    def ok(self) -> bool:
        return self.fs_sampled >= self.fs_min

    @property
    def residual_db(self) -> float:
        """Cuánto se ha reducido el aliasing respecto de no filtrar (dB)."""
        if self.energy_above <= 0:
            return float("-inf")
        return float(10 * np.log10(max(self.residual, 1e-15) / self.energy_above))

    def as_text(self) -> str:
        l1 = (f"Original: fs = {self.fs_original:.0f} Hz → Nyquist = {self.fs_original / 2:.0f} Hz.  "
              f"Ancho de banda efectivo ({self.fraction * 100:.0f} % de la energía): "
              f"f_max = {self.f_max:.0f} Hz → Nyquist-Shannon exige fs ≥ 2·f_max = {self.fs_min:.0f} Hz.")
        verdict = "CUMPLE ✔ (fs ≥ 2·f_max)" if self.ok else "NO CUMPLE ✖ (fs < 2·f_max → aliasing)"
        l2 = (f"x(n): fs pedida {self.fs_asked:.0f} Hz → fs efectiva {self.fs_sampled:.0f} Hz "
              f"(decimación ×{self.step}, Ts = {1e6 / self.fs_sampled:.1f} µs), "
              f"Nyquist = {self.nyquist_sampled:.0f} Hz → {verdict}.")
        l3 = (f"Energía del original sobre {self.nyquist_sampled:.0f} Hz: "
              f"{self.energy_above * 100:.2f} % de la energía total se pliega sobre el espectro de x(n).")
        if self.antialias:
            l4 = (f"  Con filtro antialiasing (Butterworth orden {ANTIALIAS_ORDER}, corte en "
                  f"{self.cutoff_hz:.0f} Hz): queda el {self.residual * 100:.3f} % de la energía total "
                  f"plegándose, {abs(self.residual_db):.1f} dB por debajo de lo que se quería eliminar.")
        else:
            l4 = "  Sin filtro antialiasing: ese contenido se pliega tal cual (aliasing)."
        return "\n".join((l1, l2, l3, l4))


def nyquist_report(f: np.ndarray, mag: np.ndarray, fs_original: float, fs_sampled: float,
                   step: int, antialias: bool, fs_asked: float | None = None,
                   fraction: float = 0.99) -> NyquistReport:
    fold, resid = aliasing_residual(f, mag, fs_original, fs_sampled, antialias)
    return NyquistReport(fs_original, fs_asked if fs_asked is not None else fs_sampled, fs_sampled,
                         effective_bandwidth(f, mag, fraction), fraction, step, antialias,
                         fold, resid, ANTIALIAS_CUTOFF_RATIO * fs_sampled)
