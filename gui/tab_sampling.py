"""Pestaña 2: muestreo, FFT y teorema de Nyquist-Shannon (apartado c)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from core.analysis import pick_zoom_start
from core.pipeline import extract_segment, sample_segment
from core.sampling import magnitude_spectrum, nyquist_report
from gui.plot_tab import GenreTab
from gui.plot_utils import plot_decimate, to_db


class SamplingTab(GenreTab):
    slug = "2_muestreo_fft"

    def build_controls(self) -> None:
        self.scale = tk.StringVar(value="dB")
        ttk.Label(self.controls, text="Magnitud:").pack(side="left", padx=(16, 2))
        for txt in ("dB", "lineal"):
            ttk.Radiobutton(self.controls, text=txt, value=txt, variable=self.scale,
                            command=self.render).pack(side="left")
        self.info_label = ttk.Label(self.bottom, justify="left", wraplength=1200)
        self.info_label.pack(fill="x", padx=8, pady=4)

    def draw_track(self, track, params) -> None:
        seg = extract_segment(track.audio, params)
        smp = sample_segment(seg, params)
        f, mag = magnitude_spectrum(seg.x, seg.fs)
        fn, magn = magnitude_spectrum(smp.x_n, smp.fs_eff)
        rep = nyquist_report(f, mag, seg.fs, smp.fs_eff, smp.step, smp.antialias)
        color = track.genre.color
        db = self.scale.get() == "dB"
        fig = self.figure
        fig.suptitle(track.title(), fontsize=11)
        gs = fig.add_gridspec(2, 2)

        # x(t) continua vs x(n) muestreada
        win = 0.02
        i0 = pick_zoom_start(seg.x, seg.fs, win)
        n = min(len(seg.x) - i0, int(win * seg.fs))
        t0, t1 = seg.t[i0], seg.t[i0 + n - 1]
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot((seg.t[i0:i0 + n] - t0) * 1e3, seg.x[i0:i0 + n], color="0.6", lw=1, label="x(t)")
        m = (smp.t_n >= t0) & (smp.t_n <= t1)
        markers, stems, _ = ax1.stem((smp.t_n[m] - t0) * 1e3, smp.x_n[m], basefmt=" ")
        markers.set(color=color, markersize=4)
        stems.set(color=color, linewidth=0.8)
        ax1.plot([], [], "o", color=color, ms=4, label=f"x(n), fs = {smp.fs_eff:.0f} Hz")
        ax1.set(title=f"Muestreo: x(n) = x(t = n·Ts), Ts = {1e6 / smp.fs_eff:.1f} µs", xlabel="t [ms]", ylabel="amplitud")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(alpha=0.3)

        def spectrum_axes(ax, ff, mm, fs_s, title, fmax):
            fd, md = plot_decimate(ff, to_db(mm) if db else mm)
            ax.plot(fd, md, color=color, lw=0.7)
            ax.axvline(fs_s / 2, color="#c0392b", ls="--", lw=1.2, label=f"Nyquist = {fs_s / 2:.0f} Hz")
            ax.axvline(rep.f_max, color="#1e8449", ls=":", lw=1.4, label=f"f_max (99 %) = {rep.f_max:.0f} Hz")
            ax.set(title=title, xlabel="f [Hz]", ylabel="|X(f)| [dBFS]" if db else "|X(f)|", xlim=(0, fmax))
            if db:
                ax.set_ylim(-120, 5)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)

        ax2 = fig.add_subplot(gs[1, 0])
        spectrum_axes(ax2, f, mag, seg.fs, "Espectro de magnitud de x(t) (FFT, ventana Hann)", seg.fs / 2)
        ax3 = fig.add_subplot(gs[1, 1])
        keep = f <= smp.fs_eff / 2
        fo, mo = plot_decimate(f[keep], to_db(mag[keep]) if db else mag[keep])
        ax3.plot(fo, mo, color="0.75", lw=0.7, label="x(t) original")
        spectrum_axes(ax3, fn, magn, smp.fs_eff, "Espectro de magnitud de x(n) (muestreada)", smp.fs_eff / 2)

        self.info_label.config(text=rep.as_text())
