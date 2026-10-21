"""Audios sintéticos de demostración (para probar el programa sin música real).

No reemplazan los audios reales que pide la guía: solo permiten verificar la instalación.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.io import wavfile

FS = 44100
DUR = 8.0


def _t(dur=DUR):
    return np.arange(int(dur * FS)) / FS


def _tone(f, n, harmonics=6, tau=0.25):
    t = np.arange(n) / FS
    s = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, harmonics + 1))
    return s * np.exp(-t / tau) * np.minimum(1, t / 0.005)


def _place(buf, start_s, sig):
    i = int(start_s * FS)
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[: j - i]


def _noise_burst(n, tau, rng):
    t = np.arange(n) / FS
    return rng.standard_normal(n) * np.exp(-t / tau)


def barroca():
    y = np.zeros(int(DUR * FS))
    notes = [261.6, 329.6, 392.0, 523.3, 392.0, 329.6, 293.7, 349.2, 440.0, 349.2]
    step = 60 / 70 / 2
    for i in range(int(DUR / step)):
        _place(y, i * step, _tone(notes[i % len(notes)], int(0.6 * FS), 6, 0.25))
    return y


def metal():
    rng = np.random.default_rng(1)
    y = np.zeros(int(DUR * FS))
    step = 60 / 160 / 2
    for i in range(int(DUR / step)):
        n = int(step * 1.0 * FS)
        t = np.arange(n) / FS
        chord = sum(np.sin(2 * np.pi * f * t) for f in (82.4, 123.5, 164.8))
        g = np.tanh(9 * chord) * np.exp(-t / 0.12)
        _place(y, i * step, g)
        hat = np.diff(_noise_burst(int(0.05 * FS) + 1, 0.02, rng))
        _place(y, i * step, 0.5 * hat)
    return y


def salsa_electronica():
    rng = np.random.default_rng(2)
    y = np.zeros(int(DUR * FS))
    beat = 60 / 180
    clave = [0, 1.5, 3, 5, 6]                       # clave 3-2 en corcheas (en pulsos)
    for bar in range(int(DUR / (8 * beat)) + 1):
        for c in clave:
            tt = bar * 8 * beat + c * beat
            n = int(0.08 * FS)
            tn = np.arange(n) / FS
            _place(y, tt, 0.6 * np.sin(2 * np.pi * 2500 * tn) * np.exp(-tn / 0.02))
        for k in range(8):
            _place(y, bar * 8 * beat + k * beat, 0.9 * _tone(98.0 if k % 2 == 0 else 130.8, int(0.4 * FS), 3, 0.2))
            if k % 2 == 1:
                _place(y, bar * 8 * beat + k * beat, 0.4 * _tone(440.0, int(0.25 * FS), 8, 0.1))
    return y


def cantina():
    y = np.zeros(int(DUR * FS))
    step = 60 / 100
    chords = [(220.0, 261.6, 329.6), (196.0, 246.9, 293.7), (174.6, 220.0, 261.6), (164.8, 207.7, 246.9)]
    for i in range(int(DUR / step)):
        for j, f in enumerate(chords[(i // 2) % 4]):
            _place(y, i * step + j * 0.03, 0.5 * _tone(f, int(0.9 * FS), 10, 0.4))
    t = _t()
    vib = 330 + 330 * 0.02 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum(vib) / FS
    voice = sum(np.sin(k * phase) / k for k in range(1, 5)) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t)) ** 2
    return y + 0.7 * voice


def reggaeton():
    rng = np.random.default_rng(3)
    y = np.zeros(int(DUR * FS))
    beat = 60 / 95
    for b in range(int(DUR / beat) + 1):
        n = int(0.4 * FS)
        t = np.arange(n) / FS
        f = 45 + 80 * np.exp(-t / 0.03)
        kick = np.sin(2 * np.pi * np.cumsum(f) / FS) * np.exp(-t / 0.15)
        _place(y, b * beat, 1.2 * kick)
        _place(y, b * beat + 0.75 * beat, 0.7 * _noise_burst(int(0.15 * FS), 0.05, rng))
        _place(y, b * beat + 0.375 * beat, 0.15 * np.diff(_noise_burst(int(0.05 * FS) + 1, 0.015, rng)))
        _place(y, b * beat, 0.35 * _tone(55.0, int(beat * FS), 2, 0.5))
        _place(y, b * beat + 0.5 * beat, 0.3 * np.sign(_tone(440.0, int(0.2 * FS), 1, 0.1)))
    return y


GENERATORS = {"barroca": barroca, "metal": metal, "salsa_electronica": salsa_electronica,
              "cantina": cantina, "reggaeton": reggaeton}


def write_demo_wavs(folder: str) -> dict[str, str]:
    """Escribe 5 WAV de 16 bits en `folder` y devuelve {clave_género: ruta}."""
    os.makedirs(folder, exist_ok=True)
    paths = {}
    for key, gen in GENERATORS.items():
        y = gen()
        y = 0.8 * y / np.max(np.abs(y))
        path = os.path.join(folder, f"demo_{key}.wav")
        wavfile.write(path, FS, (y * 32767).astype(np.int16))
        paths[key] = path
    return paths
