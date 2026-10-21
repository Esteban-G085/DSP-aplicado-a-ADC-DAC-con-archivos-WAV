"""Pestaña 1: señal x(t) en el tiempo, zoom y espectrograma (apartado c)."""
from __future__ import annotations

import numpy as np
from tkinter import ttk

from core.analysis import pick_zoom_start
from core.pipeline import extract_segment
from gui.plot_tab import GenreTab


class SignalTab(GenreTab):
    slug = "1_senal_xt"

    def build_controls(self) -> None:
        pass

    def draw_track(self, track, params) -> None:
        seg = extract_segment(track.audio, params)
        color = track.genre.color
        fig = self.figure
        fig.suptitle(track.title(), fontsize=11)
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(seg.t, seg.x, lw=0.5, color=color)
        ax1.set(title="x(t) — segmento analizado", xlabel="t [s]", ylabel="amplitud (escala completa = 1)")
        ax1.set_xlim(seg.t[0], seg.t[-1])
        ax1.grid(alpha=0.3)
        sec = ax1.secondary_yaxis("right", functions=(lambda a: a * params.vref, lambda v: v / params.vref))
        sec.set_ylabel(f"x(t) [V]  (±Vref = ±{params.vref:g} V)")

        win = 0.03
        i0 = pick_zoom_start(seg.x, seg.fs, win)
        n = min(len(seg.x) - i0, int(win * seg.fs))
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot((seg.t[i0:i0 + n] - seg.t[i0]) * 1e3, seg.x[i0:i0 + n], color=color, lw=1)
        ax2.set(title=f"Zoom de {win * 1e3:.0f} ms (tramo de mayor energía)", xlabel="t [ms]", ylabel="amplitud")
        ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 1])
        with np.errstate(divide="ignore"):
            _, _, _, im = ax3.specgram(seg.x, NFFT=2048, Fs=seg.fs, noverlap=1024, cmap="magma",
                                       xextent=(seg.t[0], seg.t[-1]))
        hi = im.get_clim()[1]
        im.set_clim(hi - 90, hi)
        ax3.set(title="Espectrograma", xlabel="t [s]", ylabel="f [Hz]", ylim=(0, min(seg.fs / 2, 16000)))
        fig.colorbar(im, ax=ax3, label="dB")

        rms = float(np.sqrt(np.mean(seg.x ** 2)))
        a = track.audio
        self._info(f"Archivo: {a.name} | fs = {a.fs} Hz | {a.bits} bits | N = {len(seg.x)} muestras | "
                   f"duración analizada = {len(seg.x) / seg.fs:.2f} s | pico = {np.max(np.abs(seg.x)):.3f} | "
                   f"RMS = {20 * np.log10(rms + 1e-12):.1f} dBFS | ganancia de normalización = ×{seg.gain:.2f}")

    def _info(self, text: str) -> None:
        if not hasattr(self, "info_label"):
            self.info_label = ttk.Label(self.bottom, justify="left", wraplength=1200)
            self.info_label.pack(fill="x", padx=8, pady=4)
        self.info_label.config(text=text)
