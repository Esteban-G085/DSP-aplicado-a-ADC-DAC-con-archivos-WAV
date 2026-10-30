"""Pestaña 4: perfil espectral por género y heurística TDAH (apartado e)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from core.tdah import (ENERGY_BANDS, WEIGHT_LABELS, WEIGHTS, GenreAnalysis, analyze_genre,
                       explain, rank)
from gui.plot_tab import PlotTab
from gui.plot_utils import fmt

COMPONENT_COLORS = ("#1e8449", "#2471a3", "#d68910", "#7b5ea7")
BOTTOM_HEIGHT = 300        # alto del panel de texto [px]


class TdahTab(PlotTab):
    """Panel de tres secciones sobre el análisis de frecuencias por género.

    Sección superior: nivel espectral en bandas de 1/3 de octava (eje X
    logarítmico) con una línea guía por género en su centroide espectral.
    Sección inferior izquierda: reparto de la energía en 7 bandas.
    Sección inferior derecha: índice heurístico TDAH en barras apiladas.
    """

    slug = "4_perfil_tdah"
    bottom_height = BOTTOM_HEIGHT

    def __init__(self, master, app):
        super().__init__(master, app)
        ttk.Button(self.controls, text="Graficar", command=self.render).pack(side="left")
        ttk.Label(self.controls, foreground="#555", text=(
            "  Perfil espectral por género y perfil heurístico TDAH (análisis del audio completo)."
        )).pack(side="left")
        self.text = tk.Text(self.bottom, height=17, wrap="none", font=("Consolas", 9),
                            relief="flat", background="#f4f4f4")
        self.text.pack(fill="both", expand=True, padx=8, pady=4)
        self.text.config(state="disabled")

    def _set_text(self, s: str) -> None:
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", s)
        self.text.config(state="disabled")

    def clear_text(self) -> None:
        self._set_text("")

    # --- dibujo -----------------------------------------------------------
    def draw(self) -> None:
        tracks = self.app.session.loaded()
        if not tracks:
            self.placeholder()
            self._set_text("Cargue al menos un audio WAV en la pestaña «0. Carga de audios».")
            return
        analyses = [analyze_genre(t) for t in tracks]
        ordered = rank(analyses)
        fig = self.figure
        fig.suptitle("Perfil espectral por género musical y perfil heurístico TDAH", fontsize=12)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])

        self._draw_octave(ordered, fig.add_subplot(gs[0, :]))
        self._draw_bands(analyses, fig.add_subplot(gs[1, 0]))
        self._draw_index(ordered, fig.add_subplot(gs[1, 1]))
        self._set_text(explain(analyses))

    # --- sección superior: 1/3 de octava ----------------------------------
    def _draw_octave(self, analyses: list[GenreAnalysis], ax) -> None:
        lowest = min(float(np.min(a.profile.octave_db)) for a in analyses)
        for a in analyses:
            p = a.profile
            ax.plot(p.octave_centers, p.octave_db, "-o", color=a.color, lw=1.2, ms=3,
                    label=f"{a.label}  (centroide {p.centroid_hz:.0f} Hz)")
            # Línea guía: la posición exacta del centroide espectral.
            ax.axvline(p.centroid_hz, color=a.color, ls=":", lw=1.0, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlim(20, 20000)
        ax.set_ylim(max(lowest - 5, -70), 3)
        ax.set(title="Nivel espectral en bandas de 1/3 de octava (líneas punteadas: centroide de cada género)",
               xlabel="frecuencia f [Hz]  (escala logarítmica)", ylabel="nivel relativo [dB]")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper right", ncol=1)

    # --- sección inferior izquierda: reparto de energía -------------------
    def _draw_bands(self, analyses: list[GenreAnalysis], ax) -> None:
        labels = [b[0] for b in ENERGY_BANDS]
        n = len(analyses)
        width = 0.8 / max(n, 1)
        x = np.arange(len(labels))
        for i, a in enumerate(analyses):
            offset = (i - (n - 1) / 2.0) * width
            ax.bar(x + offset, a.profile.band_pct, width=width * 0.92, color=a.color,
                   label=a.label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set(title="Distribución de energía por banda (% de la energía total)",
               xlabel="banda de frecuencia", ylabel="energía [%]")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7, loc="upper right")

    # --- sección inferior derecha: índice heurístico ----------------------
    def _draw_index(self, analyses: list[GenreAnalysis], ax) -> None:
        """Barras apiladas: cada segmento es la contribución ponderada.

        Los géneros llegan ORDENADOS por índice, de modo que la gráfica se
        lee como el ranking del bloque de texto.
        """
        x = np.arange(len(analyses))
        bottom = np.zeros(len(analyses))
        for k, color in enumerate(COMPONENT_COLORS):
            values = np.array([a.score.contributions[k] for a in analyses])
            ax.bar(x, values, bottom=bottom, width=0.62, color=color,
                   label=f"{WEIGHT_LABELS[k]} ({WEIGHTS[k] * 100:.0f} %)")
            bottom += values
        for xi, a in zip(x, analyses):
            ax.text(xi, a.score.index + 1.5, f"{a.score.index:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([a.label for a in analyses], rotation=20, ha="right", fontsize=8)
        ax.set(title="Perfil heurístico TDAH (índice 0 - 100)", xlabel="género musical",
               ylabel="índice evaluado")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7, loc="upper right")

        ax.text(0.5, -0.30, "Heurística didáctica: no es una evaluación clínica ni un diagnóstico médico.",
                transform=ax.transAxes, ha="center", va="top", fontsize=7, color="#7b241c")
