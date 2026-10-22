"""Pestaña 1: señal x(t), muestreo con FFT, Nyquist-Shannon y espectro de magnitud (apartado c)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from core.pipeline import extract_segment, sample_segment
from core.quantization import AdcConfig, quantize
from core.sampling import magnitude_spectrum, nyquist_report, pick_zoom_start
from gui.plot_tab import GenreTab
from gui.plot_utils import fmt, plot_decimate, to_db


class SamplingTab(GenreTab):
    slug = "1_muestreo_fft"

    def build_controls(self) -> None:
        self.scale = tk.StringVar(value="lineal")
        ttk.Label(self.controls, text="Magnitud:").pack(side="left", padx=(16, 2))
        for txt in ("dB", "lineal"):
            ttk.Radiobutton(self.controls, text=txt, value=txt, variable=self.scale,
                            command=self.render).pack(side="left")
        self.info_label = ttk.Label(self.bottom, justify="left", wraplength=1250)
        self.info_label.pack(fill="x", padx=8, pady=4)

    def clear_text(self) -> None:
        self.info_label.config(text="")

    def draw_track(self, track, params) -> None:
        seg = extract_segment(track.audio, params)
        smp = sample_segment(seg, params)
        cfg = AdcConfig(params.bits, params.vref)
        qr = quantize(smp.x_n, cfg)

        # El informe de Nyquist se calcula sobre la señal ORIGINAL (sin filtrar):
        # el filtro antialiasing no puede "quitar" energía que nunca estuvo.
        x_orig = seg.x_raw * seg.gain
        f, mag = magnitude_spectrum(x_orig, seg.fs)
        fn, magn = magnitude_spectrum(smp.x_n, smp.fs_eff)
        mag *= params.vref
        magn *= params.vref
        rep = nyquist_report(f, mag, seg.fs, smp.fs_eff, smp.step, smp.antialias,
                             fs_asked=params.fs_target)
        color = track.genre.color
        db = self.scale.get() == "dB"
        fig = self.figure
        fig.suptitle(f"{track.title()} — x(t) → x(n) → ADC/DAC de {cfg.bits} bits", fontsize=11)
        gs = fig.add_gridspec(2, 2)

        # --- (0,0) señal de entrada x(t) completa ------------------------
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(seg.t, x_orig, lw=0.5, color=color)
        ax0.set(title="Señal de entrada x(t) — segmento analizado", xlabel="t [s]",
                ylabel="amplitud (escala completa = 1)")
        ax0.set_xlim(seg.t[0], seg.t[-1])
        ax0.grid(alpha=0.3)
        sec = ax0.secondary_yaxis("right", functions=(lambda a: a * params.vref,
                                                      lambda v: v / params.vref))
        sec.set_ylabel(f"x(t) [V]  (±Vref = ±{params.vref:g} V)")

        # --- (0,1) x(n) muestreada + escalera del DAC (los bits) ---------
        win = 0.03
        i0 = pick_zoom_start(smp.x_n, smp.fs_eff, win)
        n = max(8, min(len(smp.x_n) - i0, int(win * smp.fs_eff)))
        sl = slice(i0, i0 + n)
        t_win = smp.t_n[i0]                       # instante inicial de la ventana
        tm = (smp.t_n[sl] - t_win) * 1e3

        # x(t) del mismo tramo (la que se muestreó, ya filtrada), en voltios
        j0 = i0 * smp.step
        j1 = min(j0 + n * smp.step, len(seg.x))
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.plot((seg.t[j0:j1] - t_win) * 1e3, seg.x[j0:j1] * params.vref,
                 color="0.6", lw=1, label="x(t) filtrada" if seg.filtered else "x(t)")
        markers, stems, _ = ax1.stem(tm, smp.x_n[sl] * params.vref, basefmt=" ")
        markers.set(color=color, markersize=4)
        stems.set(color=color, linewidth=0.8)
        ax1.plot([], [], "o", color=color, ms=4, label=f"x(n), fs = {smp.fs_eff:.0f} Hz")
        ax1.step(tm, qr.volts_q[sl], where="mid", color="#c0392b", lw=1.4,
                 label=f"salida DAC ({cfg.bits} bits, Δ = {fmt(cfg.lsb * 1e3, 4)} mV)")
        ax1.set(title=f"Muestreo x(n) = x(n·Ts) y su cuantización "
                       f"(Δ = {fmt(cfg.lsb * 1e3, 4)} mV)", xlabel="t [ms]", ylabel="V")
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
        ax1.grid(alpha=0.3)

        # --- (1,0) espectro de magnitud ----------------------------------
        ax2 = fig.add_subplot(gs[1, 0])
        fd, md = plot_decimate(f, to_db(mag) if db else mag)
        fd2, md2 = plot_decimate(fn, to_db(magn) if db else magn)
        if db:
            ax2.plot(fd, md, color="0.7", lw=0.7, label="x(t)")
            ax2.plot(fd2, md2, color=color, lw=0.9, label="x(n)")
        else:
            ax2.vlines(fd, 0, md, color="0.7", lw=0.55, alpha=0.7, label="x(t)")
            ax2.vlines(fd2, 0, md2, color=color, lw=0.75, label="x(n)")
        ax2.axvline(smp.fs_eff / 2, color="#c0392b", ls="--", lw=1.2,
                    label=f"Nyquist de x(n) = {smp.fs_eff / 2:.0f} Hz")
        ax2.axvline(rep.f_max, color="#1e8449", ls=":", lw=1.4,
                    label=f"f_max (99 %) = {rep.f_max:.0f} Hz")
        ax2.set(title="Espectro de magnitud unilateral |X(f)| (FFT, escala 2/N)", xlabel="f [Hz]",
                ylabel="|X(f)| [dBV]" if db else "Amplitud [V]", xlim=(0, seg.fs / 2))
        if db:
            ax2.set_ylim(-120, max(5, 20 * np.log10(params.vref) + 5))
        else:
            ax2.set_ylim(bottom=0)
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
        ax2.grid(alpha=0.3)

        # --- (1,1) error de cuantización ---------------------------------
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(tm, qr.error[sl] * 1e3, color="#c0392b", lw=0.9,
                 label=f"e(n) = V_DAC − x(n)  ({cfg.bits} bits)")
        for s in (+1, -1):
            ax3.axhline(s * cfg.lsb / 2 * 1e3, color="k", ls="--", lw=0.8)
        ax3.set(title="Error de cuantización e(n)   (líneas: ±Δ/2)", xlabel="t [ms]",
                ylabel="mV")
        ax3.grid(alpha=0.3)
        ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=8)

        # --- veredicto de Nyquist-Shannon ---------------------------------
        extra = ""
        if seg.truncated:
            extra = (f"\nNota: se pidieron {seg.requested_s:.1f} s pero el archivo solo tiene "
                     f"{seg.actual_s:.1f} s disponibles; se analizaron esas.")
        if seg.filtered:
            extra += (f"\nNota: antes de muestrear se aplicó el filtro antialiasing "
                      f"(corte en {rep.cutoff_hz:.0f} Hz); el pico se normalizó sobre la "
                      f"señal filtrada para que el ADC no sature.")
        self.info_label.config(text=rep.as_text() + extra)
