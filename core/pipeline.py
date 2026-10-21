"""Parámetros globales y cadena x(t) -> segmento -> x(n)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_io import Audio
from .errors import ParamsError, UserError
from .sampling import decimate

HEADROOM = 0.99   # al normalizar, el pico queda en 0.99 de la escala completa del ADC


@dataclass(frozen=True)
class Params:
    start_s: float = 0.0
    duration_s: float = 10.0
    fs_target: int = 22050
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
    x: np.ndarray        # x(t) discretizada a la fs del archivo, escala ±1
    fs: int
    t: np.ndarray        # [s] respecto al inicio del archivo
    start_s: float
    gain: float          # ganancia aplicada por la normalización


@dataclass
class Sampled:
    x_n: np.ndarray
    t_n: np.ndarray
    fs_eff: float
    step: int
    antialias: bool


def extract_segment(audio: Audio, p: Params) -> Segment:
    i0 = int(round(p.start_s * audio.fs))
    if i0 >= len(audio.x) - 1024:
        raise UserError(f"El inicio ({p.start_s:.2f} s) supera la duración del audio "
                        f"({audio.duration:.2f} s). Reduzca el inicio.")
    i1 = min(len(audio.x), i0 + int(round(p.duration_s * audio.fs)))
    x = audio.x[i0:i1].astype(np.float64)
    gain = 1.0
    if p.normalize:
        peak = float(np.max(np.abs(x)))
        if peak > 0:
            gain = HEADROOM / peak
            x = x * gain
    t = i0 / audio.fs + np.arange(len(x)) / audio.fs
    return Segment(x=x, fs=audio.fs, t=t, start_s=i0 / audio.fs, gain=gain)


def sample_segment(seg: Segment, p: Params) -> Sampled:
    step, fs_eff, x_n = decimate(seg.x, seg.fs, p.fs_target, p.antialias)
    t_n = seg.t[0] + np.arange(len(x_n)) * step / seg.fs
    return Sampled(x_n=x_n, t_n=t_n, fs_eff=fs_eff, step=step, antialias=p.antialias and step > 1)
