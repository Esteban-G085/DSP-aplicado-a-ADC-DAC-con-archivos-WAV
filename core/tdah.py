"""Perfil espectral por género y heurística TDAH (apartado e).

Este módulo es puro DSP: no importa nada de la interfaz.  Calcula, para un
audio musical, las medidas que alimentan el panel de tres secciones:

  1. nivel en bandas de 1/3 de octava (20 Hz - 20 kHz) y centroide espectral;
  2. reparto de la energía en 7 bandas anchas;
  3. un índice heurístico de 0 a 100 a partir de cuatro componentes
     ponderados (centroide bajo, pocos agudos, espectro estable y nivel
     constante).

AVISO IMPORTANTE
----------------
El índice es una HEURÍSTICA DIDÁCTICA construida a partir de los cuatro
criterios de estimulación sensorial definidos para el trabajo; no está
validado contra ningún estudio clínico ni sirve para diagnosticar el TDAH.
Las anclas y los pesos son decisiones pedagógicas, no resultados de la
literatura.  En la interfaz se muestra siempre esta aclaración.

Medidas
--------
- Centroide espectral: media de frecuencias ponderada por la potencia.
- Energía sobre 4 kHz: fracción de la energía total por encima de 4 kHz.
- Flujo espectral: distancia L2 media entre los espectros de magnitud de
  tramas consecutivas (normalizados por su propio máximo).  Mide cuánto
  cambia el espectro: valores altos = transitorios y cambios bruscos.
- Variación de nivel: desviación estándar del RMS en dB entre tramas.
  Mide cuánto cambia el volumen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .audio_io import Audio
from .pipeline import HEADROOM
from .session import Track

# --- rejilla de 1/3 de octava -------------------------------------------
# Centros preferidos ISO 266.  El borde de una banda es el centro por
# 2^(±1/6), de modo que el borde entre dos centros consecutivos coincide
# con la media geométrica de ambos.
THIRD_OCTAVE_CENTERS: tuple[float, ...] = (
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
    12500, 16000, 20000,
)

# --- las 7 bandas anchas del reparto de energía -------------------------
# Cubren 20 - 20000 Hz sin huecos ni solapes.
ENERGY_BANDS: tuple[tuple[str, float, float], ...] = (
    ("Sub-grave", 20.0, 60.0),
    ("Grave", 60.0, 250.0),
    ("Medio-grave", 250.0, 500.0),
    ("Medio", 500.0, 2000.0),
    ("Medio-agudo", 2000.0, 4000.0),
    ("Presencia", 4000.0, 6000.0),
    ("Brillo", 6000.0, 20000.0),
)

# --- anclas del índice heurístico ---------------------------------------
# Ancla = valor a partir del cual la sub-puntuación llega a 0.
CENTROID_FULL_HZ = 3000.0     # centroide >= 3000 Hz -> 0 puntos
CENTROID_BEST_HZ = 500.0      # centroide <=  500 Hz -> 100 puntos
HIGHS_FULL_PCT = 10.0         # >= 10 % de energía sobre 4 kHz -> 0 puntos
FLUX_FULL = 3.0               # flujo >= 3.0 -> 0 puntos
LEVEL_STD_FULL_DB = 8.0       # desviación >= 8 dB -> 0 puntos

WEIGHTS: tuple[float, float, float, float] = (0.30, 0.25, 0.25, 0.20)
WEIGHT_LABELS: tuple[str, ...] = (
    "Centroide bajo", "Pocos agudos >4 kHz", "Espectro estable", "Nivel constante",
)

# Tope de análisis: los audios de trabajo duran ~30 s, pero se limita para
# que una pista larga no bloquee la interfaz.
MAX_ANALYSIS_S = 60.0

FRAME_LEN = 2048               # longitud de trama para flujo y nivel [muestras]
FRAME_HOP = 1024               # solapamiento 50 %


def third_octave_edges(centers: Sequence[float] = THIRD_OCTAVE_CENTERS) -> np.ndarray:
    """Bordes de las bandas de 1/3 de octava (n + 1 bordes para n bandas)."""
    c = np.asarray(centers, dtype=np.float64)
    ratio = 2.0 ** (1.0 / 6.0)
    edges = np.empty(c.size + 1, dtype=np.float64)
    edges[0] = c[0] / ratio
    edges[1:-1] = c[:-1] * ratio
    edges[-1] = c[-1] * ratio
    return edges


def supported_bands(fs: float, centers: Sequence[float] = THIRD_OCTAVE_CENTERS) -> np.ndarray:
    """Índices de las bandas cuyo borde superior cabe bajo el Nyquist.

    A 44.1 kHz la banda de 20 kHz termina en 22.45 kHz, por encima del
    Nyquist (22.05 kHz): esa banda no se puede integrar bien y se descarta
    para no dibujar una caída artificial en la curva.
    """
    edges = third_octave_edges(centers)
    return np.flatnonzero(edges[1:] <= fs / 2.0)


@dataclass(frozen=True)
class TdahProfile:
    """Medidas espectrales y temporales de un género."""
    name: str
    fs: int
    duration_s: float
    centroid_hz: float
    pct_above_4k: float
    spectral_flux: float
    level_std_db: float
    band_pct: np.ndarray        # 7 valores: % de la energía total por banda
    octave_centers: np.ndarray  # centros de 1/3 de octava realmente usados
    octave_db: np.ndarray       # nivel de cada banda, relativo al máximo (dB)
    octave_rms_db: np.ndarray   # nivel absoluto de cada banda (dB)

    @property
    def band_labels(self) -> tuple[str, ...]:
        return tuple(b[0] for b in ENERGY_BANDS)

    @property
    def bands_above_4k(self) -> tuple[float, float]:
        """% de energía en las dos bandas por encima de 4 kHz."""
        return float(self.band_pct[5]), float(self.band_pct[6])


@dataclass(frozen=True)
class TdahScore:
    """Índice heurístico y sus cuatro componentes."""
    centroid: float      # centroide bajo (30 %)
    highs: float         # pocos agudos > 4 kHz (25 %)
    stable: float        # espectro estable (25 %)
    level: float         # nivel constante (20 %)
    index: float         # suma ponderada, 0 - 100

    @property
    def components(self) -> tuple[tuple[str, float, float], ...]:
        """(etiqueta, sub-puntuación, peso) de cada componente."""
        return tuple(
            (label, value, weight)
            for label, value, weight in zip(WEIGHT_LABELS, self.raw, WEIGHTS)
        )

    @property
    def raw(self) -> tuple[float, float, float, float]:
        return (self.centroid, self.highs, self.stable, self.level)

    @property
    def contributions(self) -> tuple[float, float, float, float]:
        """Contribución de cada componente al índice final (ya ponderada)."""
        return tuple(v * w for v, w in zip(self.raw, WEIGHTS))


@dataclass(frozen=True)
class GenreAnalysis:
    """Perfil + puntuación de un género, listo para graficar."""
    label: str
    color: str
    profile: TdahProfile
    score: TdahScore


# --- cálculo ------------------------------------------------------------
def _normalise(x: np.ndarray) -> np.ndarray:
    """Escala la señal al mismo pico que el resto de la aplicación."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        return x * (HEADROOM / peak)
    return x.astype(np.float64)


def _psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """PSD unilateral (potencia por Hz) con ventana Hann, y su eje de frecuencias."""
    n = len(x)
    w = np.hanning(n)
    spec = np.fft.rfft(x * w)
    psd = (np.abs(spec) ** 2) / (fs * float(np.sum(w ** 2)))
    return np.fft.rfftfreq(n, 1.0 / fs), psd


def _temporal_metrics(x: np.ndarray) -> tuple[float, float]:
    """Flujo espectral y variación de nivel (dB) entre tramas."""
    n = len(x)
    if n < FRAME_LEN:
        x = np.pad(x, (0, FRAME_LEN - n))
        n = FRAME_LEN
    window = np.hanning(FRAME_LEN)
    previous: np.ndarray | None = None
    diffs: list[float] = []
    levels: list[float] = []
    for start in range(0, n - FRAME_LEN + 1, FRAME_HOP):
        frame = x[start:start + FRAME_LEN]
        magnitude = np.abs(np.fft.rfft(frame * window))
        peak = float(np.max(magnitude))
        if peak > 0:
            magnitude = magnitude / peak
        if previous is not None:
            diffs.append(float(np.sqrt(np.sum((magnitude - previous) ** 2))))
        previous = magnitude
        rms = float(np.sqrt(np.mean(frame ** 2)))
        levels.append(20.0 * np.log10(rms + 1e-12))
    flux = float(np.mean(diffs)) if diffs else 0.0
    level_std = float(np.std(levels)) if levels else 0.0
    return flux, level_std


def analyze_track(audio: Audio, *, max_s: float = MAX_ANALYSIS_S) -> TdahProfile:
    """Perfil espectral de un audio.

    Se analiza el audio COMPLETO (hasta `max_s` segundos), no el segmento
    global de Inicio/Duración: las medidas de flujo espectral y de
    variación de nivel necesitan una ventana larga para estabilizarse, y
    el perfil de un género no debe depender de un corte arbitrario.
    """
    fs = int(audio.fs)
    limit = int(max_s * fs)
    x = np.asarray(audio.x[:limit], dtype=np.float64)
    if x.size < 2048:
        raise ValueError(f"El audio «{audio.name}» es demasiado corto para el análisis.")
    x = _normalise(x)

    f, psd = _psd(x, fs)
    df = fs / len(x)
    band = (f >= 20.0) & (f <= 20000.0)
    fb, pb = f[band], psd[band] * df
    total = float(np.sum(pb))
    if total <= 0:
        raise ValueError(f"El audio «{audio.name}» no tiene energía (¿está en silencio?)")

    centroid = float(np.sum(fb * pb) / total)
    pct_above_4k = 100.0 * float(np.sum(pb[fb > 4000.0]) / total)

    band_pct = np.array([
        100.0 * float(np.sum(pb[(fb >= lo) & (fb < hi)]) / total)
        for _, lo, hi in ENERGY_BANDS
    ])

    # Las bandas de 1/3 de octava se integran sobre el espectro COMPLETO: la
    # banda de 20 Hz abarca 17.82 - 22.45 Hz y su parte baja queda por debajo
    # del corte de 20 Hz usado arriba para el centroide y las 7 bandas.
    edges = third_octave_edges()
    idx = supported_bands(fs)
    centers = np.asarray(THIRD_OCTAVE_CENTERS)[idx]
    rms_db = np.array([
        20.0 * np.log10(np.sqrt(float(np.sum(psd[(f >= edges[i]) & (f < edges[i + 1])] * df))) + 1e-30)
        for i in idx
    ])
    octave_db = rms_db - float(np.max(rms_db))

    flux, level_std = _temporal_metrics(x)
    return TdahProfile(
        name=audio.name,
        fs=fs,
        duration_s=len(x) / fs,
        centroid_hz=centroid,
        pct_above_4k=pct_above_4k,
        spectral_flux=flux,
        level_std_db=level_std,
        band_pct=band_pct,
        octave_centers=centers,
        octave_db=octave_db,
        octave_rms_db=rms_db,
    )


def analyze_genre(track: Track, *, max_s: float = MAX_ANALYSIS_S) -> GenreAnalysis:
    """Perfil y puntuación de la pista cargada en una ranura de género."""
    profile = analyze_track(track.audio, max_s=max_s)
    return GenreAnalysis(
        label=track.genre.label,
        color=track.genre.color,
        profile=profile,
        score=score(profile),
    )


# --- índice heurístico ---------------------------------------------------
def _ramp(value: float, best: float, full: float, *, log: bool = False) -> float:
    """Sub-puntuación de 100 (en `best`) a 0 (en `full`), recortada a [0, 100]."""
    if log:
        if value <= best:
            return 100.0
        if value >= full:
            return 0.0
        span = np.log10(full) - np.log10(best)
        return float(100.0 * (1.0 - (np.log10(value) - np.log10(best)) / span))
    if value <= best:
        return 100.0
    if value >= full:
        return 0.0
    return float(100.0 * (1.0 - (value - best) / (full - best)))


def score(profile: TdahProfile) -> TdahScore:
    """Índice heurístico de 0 a 100 a partir de las cuatro componentes."""
    s_centroid = _ramp(profile.centroid_hz, CENTROID_BEST_HZ, CENTROID_FULL_HZ, log=True)
    s_highs = _ramp(profile.pct_above_4k, 0.0, HIGHS_FULL_PCT)
    s_stable = _ramp(profile.spectral_flux, 0.0, FLUX_FULL)
    s_level = _ramp(profile.level_std_db, 0.0, LEVEL_STD_FULL_DB)
    index = sum(v * w for v, w in zip((s_centroid, s_highs, s_stable, s_level), WEIGHTS))
    return TdahScore(s_centroid, s_highs, s_stable, s_level, float(index))


def rank(analyses: Sequence[GenreAnalysis]) -> list[GenreAnalysis]:
    """Géneros ordenados de mayor a menor índice."""
    return sorted(analyses, key=lambda a: a.score.index, reverse=True)


# --- texto explicativo ---------------------------------------------------
def explain(analyses: Sequence[GenreAnalysis]) -> str:
    """Bloque de texto del panel: ranking, detalle, criterio y nota."""
    if not analyses:
        return "Cargue al menos un audio WAV en la pestaña «0. Carga de audios»."

    ordered = rank(analyses)
    lines: list[str] = []

    lines.append("RANKING — índice heurístico TDAH (de mayor a menor)")
    lines.append("-" * 78)
    for i, a in enumerate(ordered, 1):
        s = a.score
        lines.append(f"  {i}. {a.label:<26} índice {s.index:5.1f}   "
                     f"[centroide {s.centroid:5.1f} | agudos {s.highs:5.1f} | "
                     f"estable {s.stable:5.1f} | nivel {s.level:5.1f}]")
    lines.append("")

    best, worst = ordered[0], ordered[-1]
    lines.append("DETALLE — mejor evaluado: " + best.label)
    lines.append(f"  Índice {best.score.index:.1f}  |  archivo {best.profile.name}  |  "
                 f"fs {best.profile.fs} Hz  |  {best.profile.duration_s:.1f} s analizados")
    lines.append(f"  Centroide espectral : {best.profile.centroid_hz:.0f} Hz")
    lines.append(f"  Energía sobre 4 kHz  : {best.profile.pct_above_4k:.2f} % "
                 f"(presencia {best.profile.bands_above_4k[0]:.2f} % + "
                 f"brillo {best.profile.bands_above_4k[1]:.2f} %)")
    lines.append(f"  Flujo espectral      : {best.profile.spectral_flux:.3f}")
    lines.append(f"  Variación de nivel   : {best.profile.level_std_db:.2f} dB")
    lines.append("")
    lines.append("DETALLE — peor evaluado: " + worst.label)
    lines.append(f"  Índice {worst.score.index:.1f}  |  archivo {worst.profile.name}  |  "
                 f"fs {worst.profile.fs} Hz  |  {worst.profile.duration_s:.1f} s analizados")
    lines.append(f"  Centroide espectral : {worst.profile.centroid_hz:.0f} Hz")
    lines.append(f"  Energía sobre 4 kHz  : {worst.profile.pct_above_4k:.2f} % "
                 f"(presencia {worst.profile.bands_above_4k[0]:.2f} % + "
                 f"brillo {worst.profile.bands_above_4k[1]:.2f} %)")
    lines.append(f"  Flujo espectral      : {worst.profile.spectral_flux:.3f}")
    lines.append(f"  Variación de nivel   : {worst.profile.level_std_db:.2f} dB")
    lines.append("")

    lines.append("CRITERIO — heurística de estimulación sensorial")
    lines.append(f"  1. Centroide bajo ({WEIGHTS[0] * 100:.0f} %): energía concentrada en frecuencias bajas. "
                 f"Escala logarítmica de {CENTROID_BEST_HZ:.0f} Hz (100 pts) a {CENTROID_FULL_HZ:.0f} Hz (0 pts).")
    lines.append(f"  2. Pocos agudos >4 kHz ({WEIGHTS[1] * 100:.0f} %): poca energía en presencia y brillo. "
                 f"De 0 % (100 pts) a {HIGHS_FULL_PCT:.0f} % (0 pts).")
    lines.append(f"  3. Espectro estable ({WEIGHTS[2] * 100:.0f} %): pocas mudanças bruscas entre tramas. "
                 f"Flujo espectral de 0 (100 pts) a {FLUX_FULL:.1f} (0 pts).")
    lines.append(f"  4. Nivel constante ({WEIGHTS[3] * 100:.0f} %): volumen estable. "
                 f"Desviación de 0 dB (100 pts) a {LEVEL_STD_FULL_DB:.0f} dB (0 pts).")
    lines.append("  Índice = suma ponderada de las cuatro componentes (0 - 100).")
    lines.append("")

    lines.append("NOTA")
    lines.append("  Este índice es una heurística DIDÁCTICA para el trabajo de comunicaciones: los pesos y")
    lines.append("  las anclas son decisiones pedagógicas, no resultados de la literatura, y no se han validado")
    lines.append("  contra ningún estudio clínico. Describe cómo se reparte la energía de la música, NO una")
    lines.append("  evaluación de la atención de quien la escucha y NO constituye un diagnóstico médico.")
    lines.append("  Además, cada género es una obra concreta: dos piezas del mismo género pueden puntuar distinto.")
    return "\n".join(lines)
