"""Lectura de archivos WAV (apartado a)."""
from __future__ import annotations

import os
import struct
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile

from .errors import AudioError


@dataclass
class Audio:
    path: str
    fs: int
    x: np.ndarray        # mono, float32, escala completa = ±1.0
    channels: int
    bits: int            # profundidad de bits del archivo original
    is_float: bool

    @property
    def duration(self) -> float:
        return len(self.x) / self.fs

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def _read_fmt_chunk(path: str) -> tuple[int, int, int, int]:
    """Devuelve (formato, canales, fs, bits) leyendo el chunk 'fmt ' del WAV."""
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
            raise AudioError("El archivo no es un WAV válido (cabecera RIFF/WAVE).")
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid, size = struct.unpack("<4sI", hdr)
            if cid == b"fmt ":
                tag, ch, fs, _, _, bits = struct.unpack("<HHIIHH", f.read(size)[:16])
                return tag, ch, fs, bits
            f.seek(size + (size & 1), 1)
    raise AudioError("El WAV no contiene el chunk 'fmt '.")


def _to_float(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.uint8:                       # PCM 8 bits es sin signo
        return (data.astype(np.float64) - 128.0) / 128.0
    if data.dtype.kind == "i":                       # PCM 16/24/32 (24 llega justificado a 32)
        return data.astype(np.float64) / (np.iinfo(data.dtype).max + 1.0)
    if data.dtype.kind == "f":
        return data.astype(np.float64)
    raise AudioError(f"Tipo de muestra no soportado: {data.dtype}")


def load_wav(path: str) -> Audio:
    """Carga un WAV, lo convierte a mono (promedio de canales) y a ±1.0."""
    try:
        _, _, _, bits = _read_fmt_chunk(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fs, data = wavfile.read(path)
    except AudioError:
        raise
    except Exception as exc:                         # noqa: BLE001
        raise AudioError(f"No se pudo leer el WAV: {exc}") from exc

    x = _to_float(data)
    channels = 1 if x.ndim == 1 else x.shape[1]
    if x.ndim == 2:
        x = x.mean(axis=1)
    if len(x) < 2048:
        raise AudioError("El audio es demasiado corto (mínimo 2048 muestras).")
    return Audio(path=path, fs=int(fs), x=x.astype(np.float32), channels=channels,
                 bits=int(bits), is_float=data.dtype.kind == "f")
