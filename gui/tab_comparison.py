"""Pestaña 4: frecuencias de cada género en una misma gráfica y criterio TDAH (apartado e)."""
from __future__ import annotations

import numpy as np
import tkinter as tk
from tkinter import ttk

from core.analysis import (BANDS, SCORE_LABELS, SCORE_WEIGHTS, averaged_spectrum, build_conclusion,
                           fractional_octave_spectrum)
from core.pipeline import extract_segment
from core.report import analyze_tracks
from gui.plot_tab import PlotTab

PART_COLORS = {"centroide": "#4c78a8", "agudos": "#f58518", "flujo": "#54a24b", "dinamica": "#b279a2"}


class ComparisonTab(PlotTab):
    slug = "4_comparacion_frecuencias"

    def __init__(self, master, app):
        super().__init__(master, app)
        ttk.Button(self.controls, text="Graficar", command=self.render).pack(side="left")
        ttk.Label(self.controls, foreground="#555", text="  Compara en la misma gráfica todos los géneros cargados.").pack(side="left")
        self.text = tk.Text(self.bottom, height=7, wrap="word", font=("TkDefaultFont", 10), relief="flat",
                            background="#f4f4f4")
        self.text.pack(fill="x", padx=8, pady=4)
        self.text.config(state="disabled")

    def _set_text(self, s: str) -> None:
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", s)
        self.text.config(state="disabled")

    def draw(self) -> None:
        tracks = self.app.session.loaded()
        if not tracks:
            self.placeholder()
            self._set_text("")
            return
        params = self.app.read_params()
        metrics, ranking = analyze_tracks(tracks, params)
        fig = self.figure
        fig.suptitle("Comparación de frecuencias por género musical", fontsize=12)
        gs = fig.add_gridspec(2, 2)

        # 1) Espectros superpuestos (1/3 de octava, nivel relativo a la potencia total)
        ax1 = fig.add_subplot(gs[0, :])
        for t, m in zip(tracks, metrics):
            seg = extract_segment(t.audio, params)
            f, p = averaged_spectrum(seg.x, seg.fs)
            c, lv = fractional_octave_spectrum(f, p, 3)
            ax1.plot(c, lv, lw=1.8, color=t.genre.color, label=f"{t.genre.label} (centroide {m.centroid_hz:.0f} Hz)")
            ax1.axvline(m.centroid_hz, color=t.genre.color, ls=":", lw=1)
        ax1.set(xscale="log", xlim=(31.5, 20000), xlabel="f [Hz] (escala logarítmica)",
                ylabel="nivel relativo [dB] (1/3 de octava)",
                title="Espectro de magnitud de todos los géneros (líneas punteadas: centroide espectral)")
        ax1.grid(alpha=0.3, which="both")
        ax1.legend(fontsize=8, loc="lower left")

        # 2) Energía por banda
        ax2 = fig.add_subplot(gs[1, 0])
        names = [b[0] for b in BANDS]
        w = 0.8 / len(metrics)
        x = np.arange(len(names))
        for i, (t, m) in enumerate(zip(tracks, metrics)):
            ax2.bar(x + i * w - 0.4 + w / 2, [m.band_ratio[n] * 100 for n in names], w,
                    color=t.genre.color, label=t.genre.label)
        ax2.set_xticks(x, [f"{n}\n{lo}–{hi}" for n, lo, hi in BANDS], fontsize=7)
        ax2.set(ylabel="% de la energía", title="Distribución de energía por banda [Hz]")
        ax2.grid(alpha=0.3, axis="y")

        # 3) Índice TDAH desglosado
        ax3 = fig.add_subplot(gs[1, 1])
        labels = [r.label for r in ranking]
        bottom = np.zeros(len(ranking))
        for k in SCORE_WEIGHTS:
            vals = np.array([r.parts[k] for r in ranking])
            ax3.bar(range(len(ranking)), vals, bottom=bottom, color=PART_COLORS[k],
                    label=f"{SCORE_LABELS[k]} ({SCORE_WEIGHTS[k] * 100:.0f} %)")
            bottom += vals
        for i, r in enumerate(ranking):
            ax3.text(i, r.score + 1, f"{r.score:.0f}", ha="center", fontsize=9, fontweight="bold")
        ax3.set_xticks(range(len(ranking)), labels, fontsize=7, rotation=12)
        ax3.set(ylim=(0, 112), ylabel="índice 0-100", title="Perfil frecuencial favorable para TDAH (heurística)")
        ax3.legend(fontsize=7, loc="upper right", ncol=2)
        ax3.grid(alpha=0.3, axis="y")

        self._set_text(build_conclusion(ranking, metrics))
