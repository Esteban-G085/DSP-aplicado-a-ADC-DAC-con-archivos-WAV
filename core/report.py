"""Análisis conjunto de las pistas cargadas y exportación de métricas."""
from __future__ import annotations

import csv

from .analysis import BANDS, TrackMetrics, ScoreRow, compute_metrics, rank_for_attention
from .pipeline import Params, extract_segment
from .session import Track


def analyze_tracks(tracks: list[Track], params: Params):
    """Devuelve (métricas por pista, ranking ordenado) usando el segmento global."""
    metrics = []
    for t in tracks:
        seg = extract_segment(t.audio, params)
        metrics.append(compute_metrics(t.key, t.genre.label, seg.x, seg.fs))
    return metrics, rank_for_attention(metrics)


def export_csv(path: str, tracks: list[Track], metrics: list[TrackMetrics],
               ranking: list[ScoreRow]) -> None:
    score = {r.key: r.score for r in ranking}
    band_names = [b[0] for b in BANDS]
    header = (["Género", "Músico/Agrupación", "Obra", "Año", "Archivo", "fs [Hz]",
               "Frecuencia dominante [Hz]", "Centroide [Hz]", "f95 [Hz]", "f99 [Hz]",
               "Energía >4 kHz [%]", "Flujo espectral", "Variación RMS", "RMS [dBFS]",
               "Índice TDAH (0-100)"] + [f"Banda {n} [%]" for n in band_names] + ["Investigación"])
    by_key = {t.key: t for t in tracks}
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(header)
        for m in metrics:
            t = by_key[m.key]
            w.writerow([m.label, t.meta.artist, t.meta.work, t.meta.year, t.audio.name, m.fs,
                        f"{m.dominant_hz:.1f}", f"{m.centroid_hz:.1f}", f"{m.f95_hz:.0f}", f"{m.f99_hz:.0f}",
                        f"{m.hf_ratio * 100:.2f}", f"{m.flux:.3f}", f"{m.rms_cv:.3f}", f"{m.rms_dbfs:.1f}",
                        f"{score[m.key]:.1f}"] + [f"{m.band_ratio[n] * 100:.2f}" for n in band_names]
                       + [t.meta.notes])
