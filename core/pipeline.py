"""Parámetros globales y cadena x(t) -> segmento -> x(n)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_io import Audio
from .errors import ParamsError, UserError
from .sampling import decimate, decimation_step, lowpass_antialias

HEADROOM = 0.99   # al normalizar, el pico queda en 0.99 de la escala completa del ADC


@dataclass(frozen=True)
class Params:
    start_s: float = 0.0
    duration_s: float = 10.0
    fs_target: int = 22050      # fs objetivo de x(n); la efectiva es fs/step (decimación entera)
    antialias: bool = True
    bits: int = 8
    vref: float = 1.0          # ADC/DAC bipolar: rango de entrada ±vref [V]
    normalize: bool = True

    def validate(self) -> "Params":
        if self.start_s < 0:
            raise ParamsError("El inicio del segmento no puede ser negativo.")
        if not 0.5 <= self.duration_s <= 60:
            raise ParamsError("La duración debe estar entre 0.5 y 60 s (audios cortos).")
        if self.fs_target < 200:
            raise ParamsError("La frecuencia de muestreo de x(n) debe ser ≥ 200 Hz.")
        if not 2 <= self.bits <= 24:
            raise ParamsError("Los bits del ADC/DAC deben estar entre 2 y 24.")
        if self.vref <= 0:
            raise ParamsError("Vref debe ser mayor que 0 V.")
        return self


@dataclass
class Segment:
    x: np.ndarray        # señal que entra al ADC (filtrada y normalizada)
    x_raw: np.ndarray    # segmento original, sin filtrar ni normalizar
    fs: int
    t: np.ndarray        # [s] respecto al inicio del archivo
    start_s: float
    gain: float          # ganancia aplicada por la normalización
    requested_s: float   # duración pedida por el usuario
    filtered: bool       # True si se aplicó el paso bajo antialiasing

    @property
    def actual_s(self) -> float:
        """Duración realmente disponible en el archivo (puede ser menor que la pedida)."""
        return len(self.x) / self.fs

    @property
    def truncated(self) -> bool:
        return self.actual_s < self.requested_s - 1e-9


@dataclass
class Sampled:
    x_n: np.ndarray
    t_n: np.ndarray
    fs_eff: float
    step: int
    antialias: bool


def extract_segment(audio: Audio, p: Params, *, antialias: bool | None = None) -> Segment:
    """Extrae el segmento de audio y lo deja en la escala del ADC.

    Orden de las operaciones (importante):
      1. recorte del segmento;
      2. filtro antialiasing a la fs del archivo, si después se va a decimar;
      3. normalización del pico a HEADROOM.

    El filtro va ANTES de normalizar porque un paso bajo puede sobreimpulsar en
    los transitorios (hasta un 12 % en audio real): si se normalizara antes, el
    pico ya filtrado superaría ±Vref y el ADC saturaría.  Con este orden el pico
    que entra al ADC es exactamente HEADROOM·Vref.

    `antialias=None` usa el valor de los parámetros; con `antialias=False` se
    deja el segmento sin filtrar (lo usa la comparación de espectros).
    """
    i0 = int(round(p.start_s * audio.fs))
    if i0 >= len(audio.x) - 1024:
        raise UserError(f"El inicio ({p.start_s:.2f} s) supera la duración del audio "
                        f"({audio.duration:.2f} s). Reduzca el inicio.")
    i1 = min(len(audio.x), i0 + int(round(p.duration_s * audio.fs)))
    x_raw = audio.x[i0:i1].astype(np.float64)

    step = decimation_step(audio.fs, p.fs_target)
    use_aa = p.antialias if antialias is None else antialias
    filtered = bool(use_aa and step > 1)
    x = lowpass_antialias(x_raw, audio.fs, audio.fs / step) if filtered else x_raw.copy()

    gain = 1.0
    if p.normalize:
        peak = float(np.max(np.abs(x)))
        if peak > 0:
            gain = HEADROOM / peak
            x = x * gain
    t = i0 / audio.fs + np.arange(len(x)) / audio.fs
    return Segment(x=x, x_raw=x_raw, fs=audio.fs, t=t, start_s=i0 / audio.fs, gain=gain,
                   requested_s=p.duration_s, filtered=filtered)


def sample_segment(seg: Segment, p: Params) -> Sampled:
    """Decimación: x(n) = x(n·Ts) tomando 1 de cada `step` muestras."""
    step, fs_eff, x_n = decimate(seg.x, seg.fs, p.fs_target)
    t_n = seg.t[0] + np.arange(len(x_n)) * step / seg.fs
    return Sampled(x_n=x_n, t_n=t_n, fs_eff=fs_eff, step=step, antialias=seg.filtered)
