"""Pestaña 5: todas las señales x(t) integradas en una figura."""
from __future__ import annotations

import numpy as np
from tkinter import ttk

from core.pipeline import extract_segment
from gui.plot_tab import PlotTab


class IntegratedTab(PlotTab):
    slug = "5_integrado_xt"

    def __init__(self, master, app):
        super().__init__(master, app)
        ttk.Button(self.controls, text="Graficar", command=self.render).pack(side="left")

    def draw(self) -> None:
        tracks = self.app.session.loaded()
        if not tracks:
            self.placeholder()
            return
        params = self.app.read_params()
        fig = self.figure
        fig.suptitle("Señales x(t) de todos los géneros", fontsize=12)
        gs = fig.add_gridspec(len(tracks) + 1, 1, height_ratios=[1] * len(tracks) + [1.4])
        axes = []
        env_data = []
        for i, t in enumerate(tracks):
            seg = extract_segment(t.audio, params)
            ax = fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None)
            ax.plot(seg.t, seg.x, lw=0.4, color=t.genre.color)
            ax.set_ylabel(t.genre.label, fontsize=8, rotation=0, ha="right", va="center")
            ax.set_ylim(-1.05, 1.05)
            ax.grid(alpha=0.3)
            ax.tick_params(labelbottom=False)
            axes.append(ax)
            n = max(1, int(0.05 * seg.fs))
            k = len(seg.x) // n
            rms = np.sqrt(np.mean(seg.x[: k * n].reshape(k, n) ** 2, axis=1))
            env_data.append((t, seg.t[0] + (np.arange(k) + 0.5) * n / seg.fs, 20 * np.log10(rms + 1e-6)))
        ax = fig.add_subplot(gs[len(tracks), 0], sharex=axes[0])
        for t, tt, db in env_data:
            ax.plot(tt, db, lw=1.3, color=t.genre.color, label=t.genre.label)
        ax.set(xlabel="t [s]", ylabel="nivel RMS [dBFS]", ylim=(-70, 3),
               title="Envolventes de nivel superpuestas (tramos de 50 ms)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=len(tracks), loc="lower center")
