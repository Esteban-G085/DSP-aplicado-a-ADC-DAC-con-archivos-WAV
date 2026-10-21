"""Frecuencias por género y criterio comparativo para TDAH (apartado e).

El índice es una HEURÍSTICA DIDÁCTICA, no un criterio clínico: ordena los géneros
cargados entre sí según cuatro métricas calculadas con FFT (menor = mejor en las cuatro):
    centroide espectral, energía > 4 kHz, flujo espectral y variación del nivel RMS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .sampling import cumulative_frequency

BANDS = (
    ("Sub-grave", 20, 60), ("Grave", 60, 250), ("Medio-grave", 250, 500),
    ("Medio", 500, 2000), ("Medio-agudo", 2000, 4000), ("Presencia", 4000, 6000),
    ("Brillo", 6000, 20000),
)
HF_LIMIT_HZ = 4000.0
# Pesos del índice (suman 1). Se pueden ajustar y justificar en la presentación.
SCORE_WEIGHTS = {"centroide": 0.30, "agudos": 0.25, "flujo": 0.25, "dinamica": 0.20}
SCORE_LABELS = {"centroide": "Centroide bajo", "agudos": "Pocos agudos (>4 kHz)",
                "flujo": "Espectro estable", "dinamica": "Nivel constante"}


@dataclass
class TrackMetrics:
    key: str
    label: str
    fs: int
    dominant_hz: float
    centroid_hz: float
    f95_hz: float
    f99_hz: float
    band_ratio: dict
    hf_ratio: float
    flux: float
    rms_cv: float
    rms_dbfs: float


@dataclass
class ScoreRow:
    key: str
    label: str
    score: float      # 0-100
    parts: dict       # puntos aportados por cada criterio


def averaged_spectrum(x: np.ndarray, fs: float, nperseg: int = 16384):
    """Espectro de potencia promediado (FFT con ventana Hann, 50 % de solape)."""
    n = min(nperseg, len(x))
    return signal.welch(x, fs=fs, window="hann", nperseg=n, noverlap=n // 2, scaling="spectrum")


def dominant_frequency(f, p, fmin: float = 20.0) -> float:
    m = f >= fmin
    return float(f[m][np.argmax(p[m])]) if m.any() else 0.0


def spectral_centroid(f, p, fmin: float = 20.0, fmax: float = 20000.0) -> float:
    m = (f >= fmin) & (f <= fmax)
    tot = p[m].sum()
    return float((f[m] * p[m]).sum() / tot) if tot > 0 else 0.0


def band_ratios(f, p) -> dict:
    fmax = f[-1]
    tot = p[(f >= 20) & (f <= min(20000, fmax))].sum()
    out = {}
    for name, lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out[name] = float(p[m].sum() / tot) if tot > 0 else 0.0
    return out


def spectral_flux(x: np.ndarray, fs: float) -> float:
    """Cambio medio entre espectros consecutivos (normalizados): 0 = estable."""
    _, _, Z = signal.stft(x, fs=fs, nperseg=2048, noverlap=1024)
    mag = np.abs(Z)
    mag = mag / (np.linalg.norm(mag, axis=0, keepdims=True) + 1e-12)
    if mag.shape[1] < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(mag, axis=1), axis=0)))


def rms_variation(x: np.ndarray, fs: float, frame_s: float = 0.05) -> float:
    """Coeficiente de variación del RMS por tramos de 50 ms."""
    n = max(1, int(frame_s * fs))
    k = len(x) // n
    if k < 2:
        return 0.0
    rms = np.sqrt(np.mean(x[: k * n].reshape(k, n) ** 2, axis=1))
    return float(np.std(rms) / rms.mean()) if rms.mean() > 0 else 0.0


def fractional_octave_spectrum(f, p, fraction: int = 3, fmin: float = 31.5, fmax: float = 20000.0):
    """Potencia por bandas de 1/fraction de octava -> (centros, nivel relativo en dB)."""
    fmax = min(fmax, f[-1])
    k = int(np.floor(fraction * np.log2(fmax / fmin))) + 1
    centers = fmin * 2.0 ** (np.arange(k) / fraction)
    centers = centers[centers <= fmax]
    lo = centers * 2.0 ** (-1.0 / (2 * fraction))
    hi = centers * 2.0 ** (1.0 / (2 * fraction))
    power = np.array([p[(f >= a) & (f < b)].sum() for a, b in zip(lo, hi)])
    total = power.sum()
    with np.errstate(divide="ignore"):
        level = 10 * np.log10(power / total) if total > 0 else np.full_like(power, -120.0)
    level[~np.isfinite(level)] = np.nan
    return centers, level


def pick_zoom_start(x: np.ndarray, fs: float, win_s: float) -> int:
    """Índice de inicio de la ventana de win_s con más energía (para los zooms)."""
    n = int(win_s * fs)
    if len(x) <= n:
        return 0
    c = np.concatenate(([0.0], np.cumsum(x.astype(np.float64) ** 2)))
    starts = np.arange(0, len(x) - n + 1, max(1, n // 2))
    return int(starts[np.argmax(c[starts + n] - c[starts])])


def compute_metrics(key: str, label: str, x: np.ndarray, fs: int) -> TrackMetrics:
    f, p = averaged_spectrum(x, fs)
    hf_mask = (f >= HF_LIMIT_HZ) & (f <= min(20000, f[-1]))
    tot = p[(f >= 20) & (f <= min(20000, f[-1]))].sum()
    return TrackMetrics(
        key=key, label=label, fs=fs,
        dominant_hz=dominant_frequency(f, p),
        centroid_hz=spectral_centroid(f, p),
        f95_hz=cumulative_frequency(f, p, 0.95),
        f99_hz=cumulative_frequency(f, p, 0.99),
        band_ratio=band_ratios(f, p),
        hf_ratio=float(p[hf_mask].sum() / tot) if tot > 0 else 0.0,
        flux=spectral_flux(x, fs),
        rms_cv=rms_variation(x, fs),
        rms_dbfs=float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)),
    )


def rank_for_attention(metrics: list[TrackMetrics]) -> list[ScoreRow]:
    """Índice 0-100 (mayor = perfil más favorable), relativo al conjunto cargado."""
    raw = {
        "centroide": np.array([m.centroid_hz for m in metrics]),
        "agudos": np.array([m.hf_ratio for m in metrics]),
        "flujo": np.array([m.flux for m in metrics]),
        "dinamica": np.array([m.rms_cv for m in metrics]),
    }
    fav = {}
    for name, v in raw.items():
        span = v.max() - v.min()
        fav[name] = 1.0 - (v - v.min()) / span if span > 1e-12 else np.full(len(v), 0.5)
    rows = []
    for i, m in enumerate(metrics):
        parts = {k: 100.0 * SCORE_WEIGHTS[k] * float(fav[k][i]) for k in SCORE_WEIGHTS}
        rows.append(ScoreRow(m.key, m.label, sum(parts.values()), parts))
    return sorted(rows, key=lambda r: r.score, reverse=True)


def build_conclusion(rows: list[ScoreRow], metrics: list[TrackMetrics]) -> str:
    by_key = {m.key: m for m in metrics}

    def desc(r: ScoreRow) -> str:
        m = by_key[r.key]
        return (f"{r.label}: centroide {m.centroid_hz:.0f} Hz, {m.hf_ratio * 100:.1f} % de la energía "
                f"sobre 4 kHz, flujo espectral {m.flux:.2f}, variación de nivel {m.rms_cv:.2f}")

    lines = ["Ranking (índice 0-100, mayor = perfil frecuencial más favorable entre los géneros cargados): "
             + " > ".join(f"{r.label} ({r.score:.0f})" for r in rows)]
    lines.append("Mejor perfil → " + desc(rows[0]) + ".")
    if len(rows) > 1:
        lines.append("Menos favorable → " + desc(rows[-1]) + ".")
    lines.append("Criterio: energía concentrada en graves/medios, pocos agudos y espectro y nivel estables "
                 "implican menor estimulación sensorial y menos eventos que capten la atención.")
    lines.append("Nota: es una heurística didáctica basada en la gráfica, no un diagnóstico; la respuesta de un "
                 "estudiante con TDAH a la música es individual y la evidencia científica es mixta.")
    return "\n".join(lines)
