"""Pestaña 3: comparación de los espectros de magnitud de todos los géneros en una sola gráfica."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from core.pipeline import extract_segment
from core.sampling import cumulative_frequency, dominant_frequency, magnitude_spectrum, spectral_centroid
from gui.plot_tab import PlotTab
from gui.plot_utils import plot_decimate

FMIN = 20.0


class ComparisonTab(PlotTab):
    slug = "3_comparacion_espectros"
    bottom_height = 165          # alto del panel con la tabla de métricas [px]

    def __init__(self, master, app):
        super().__init__(master, app)
        ttk.Button(self.controls, text="Graficar", command=self.render).pack(side="left")
        ttk.Label(self.controls, foreground="#555",
                  text="  Compara en la misma gráfica los espectros de magnitud de los géneros cargados."
                  ).pack(side="left")
        self.text = tk.Text(self.bottom, height=9, wrap="none", font=("Consolas", 9),
                            relief="flat", background="#f4f4f4")
        self.text.pack(fill="x", padx=8, pady=4)
        self.text.config(state="disabled")

    def _set_text(self, s: str) -> None:
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", s)
        self.text.config(state="disabled")

    def clear_text(self) -> None:
        self._set_text("")

    def draw(self) -> None:
        tracks = self.app.session.loaded()
        if not tracks:
            self.placeholder()
            return
        params = self.app.read_params()
        fig = self.figure
        fig.suptitle("Comparación de los espectros de magnitud por género musical", fontsize=12)
        ax = fig.add_subplot(111)

        rows = []
        fmax = FMIN
        for t in tracks:
            # Sin filtro antialiasing: aquí se comparan los géneros reales, no el
            # resultado de una posible decimación.
            seg = extract_segment(t.audio, params, antialias=False)
            f, mag = magnitude_spectrum(seg.x, seg.fs)
            mag *= params.vref
            p = mag ** 2
            fd, md = plot_decimate(f, mag)
            ax.plot(fd, md, lw=1.1, color=t.genre.color,
                    label=f"{t.genre.label}  (f_dom {dominant_frequency(f, p):.0f} Hz)")
            fmax = max(fmax, min(20000.0, seg.fs / 2))

            rows.append((
                t.genre.label,
                t.audio.name,
                f"{t.audio.fs}",
                f"{dominant_frequency(f, p):.0f}",
                f"{spectral_centroid(f, p):.0f}",
                f"{cumulative_frequency(f, p, 0.95):.0f}",
                f"{cumulative_frequency(f, p, 0.99):.0f}",
            ))

        ax.set(xlim=(0, fmax), xlabel="f [Hz]",
               ylabel="Amplitud [V]",
               title="Espectro unilateral de amplitud por género (FFT, escala 2/N)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        head = f"{'Género':<26}{'Archivo':<24}{'fs':>7}{'f_dom':>8}{'centroide':>11}{'f95':>8}{'f99':>8}"
        lines = [head, "-" * len(head)]
        lines += [f"{g:<26}{a:<24}{fs:>7}{fd:>8}{c:>11}{f95:>8}{f99:>8}" for g, a, fs, fd, c, f95, f99 in rows]
        lines.append("")
        lines.append("f_dom = frecuencia dominante; centroide = media de frecuencias ponderada por la potencia;")
        lines.append("f95 / f99 = frecuencia por debajo de la que se acumula el 95 % / 99 % de la energía.")
        lines.append(f"Las señales están normalizadas al mismo pico (0.99) y expresadas en voltios (Vref = {params.vref:g} V),")
        lines.append("así que los espectros son comparables en amplitud.")
        self._set_text("\n".join(lines))
