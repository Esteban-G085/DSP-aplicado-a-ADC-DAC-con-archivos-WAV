"""Pestaña 3: cuantización, ADC y DAC sobre x(n) (apartado d)."""
from __future__ import annotations

import numpy as np
from tkinter import ttk

from core.analysis import pick_zoom_start
from core.pipeline import extract_segment, sample_segment
from core.quantization import AdcConfig, quantize, sample_table, sqnr_vs_bits
from gui.plot_tab import GenreTab

COLUMNS = (("n", 50), ("t [ms]", 80), ("x(n) [norm]", 95), ("x(n) [V]", 95), ("q(n) con signo", 100),
           ("código ADC (bin)", 190), ("V_DAC [V]", 95), ("error [mV]", 90))


class QuantizationTab(GenreTab):
    slug = "3_cuantizacion_adc_dac"

    def build_controls(self) -> None:
        self.info_label = ttk.Label(self.bottom, justify="left", wraplength=1250)
        self.info_label.pack(fill="x", padx=8, pady=(4, 2))
        frame = ttk.Frame(self.bottom)
        frame.pack(fill="x", padx=8, pady=(0, 6))
        self.table = ttk.Treeview(frame, columns=[c[0] for c in COLUMNS], show="headings", height=6)
        for name, width in COLUMNS:
            self.table.heading(name, text=name)
            self.table.column(name, width=width, anchor="center")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=sb.set)
        self.table.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")

    def draw_track(self, track, params) -> None:
        seg = extract_segment(track.audio, params)
        smp = sample_segment(seg, params)
        cfg = AdcConfig(params.bits, params.vref)
        qr = quantize(smp.x_n, cfg)
        color = track.genre.color
        fig = self.figure
        fig.suptitle(f"{track.title()} — ADC/DAC de {cfg.bits} bits, ±{cfg.vref:g} V", fontsize=11)
        gs = fig.add_gridspec(2, 2)

        win = 0.02
        i0 = pick_zoom_start(smp.x_n, smp.fs_eff, win)
        n = max(8, min(len(smp.x_n) - i0, int(win * smp.fs_eff)))
        sl = slice(i0, i0 + n)
        tm = (smp.t_n[sl] - smp.t_n[i0]) * 1e3

        ax1 = fig.add_subplot(gs[0, 0])
        v, vq = qr.volts[sl], qr.volts_q[sl]
        k0, k1 = int(np.floor(v.min() / cfg.lsb)), int(np.ceil(v.max() / cfg.lsb))
        if k1 - k0 <= 48:
            for k in range(k0, k1 + 1):
                ax1.axhline(k * cfg.lsb, color="0.88", lw=0.5, zorder=0)
        ax1.plot(tm, v, "o-", color="0.55", ms=3, lw=0.8, label="x(n) [V]")
        ax1.step(tm, vq, where="mid", color=color, lw=1.4, label="salida DAC (cuantizada)")
        ax1.set(title=f"x(n) y cuantización (Δ = {cfg.lsb * 1e3:.3f} mV)", xlabel="t [ms]", ylabel="V")
        ax1.legend(fontsize=8, loc="upper right")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(tm, qr.error[sl] * 1e3, color="#c0392b", lw=0.9)
        for s in (+1, -1):
            ax2.axhline(s * cfg.lsb / 2 * 1e3, color="k", ls="--", lw=0.8)
        ax2.set(title="Error de cuantización e(n) = V_DAC − x(n)   (líneas: ±Δ/2)", xlabel="t [ms]", ylabel="mV")
        ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        bl = list(range(2, 25))
        ax3.plot(bl, sqnr_vs_bits(smp.x_n, cfg.vref, bl), "o-", ms=3, color=color, label="medida en esta señal")
        ax3.plot(bl, [6.02 * b + 1.76 for b in bl], "k--", lw=1, label="senoide a plena escala: 6.02·N + 1.76")
        ax3.axvline(cfg.bits, color="#c0392b", lw=1.2)
        ax3.plot([cfg.bits], [qr.sqnr_db], "s", color="#c0392b", label=f"N = {cfg.bits}: {qr.sqnr_db:.1f} dB")
        ax3.set(title="Relación señal-ruido de cuantización (SQNR) vs bits", xlabel="N [bits]", ylabel="SQNR [dB]")
        ax3.legend(fontsize=8, loc="upper left")
        ax3.grid(alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(qr.codes_u, bins=min(cfg.levels, 128), range=(-0.5, cfg.levels - 0.5), color=color)
        ax4.set(title="Uso del rango del ADC (histograma de códigos)", xlabel=f"código ADC (0 … {cfg.levels - 1})",
                ylabel="muestras")
        ax4.set_yscale("log")

        clip = f" | {qr.clipped} muestras saturadas" if qr.clipped else ""
        self.info_label.config(text=(
            f"N = {cfg.bits} bits → {cfg.levels} niveles | Resolución (LSB) Δ = 2·Vref/2^N = {cfg.lsb * 1e3:.4f} mV "
            f"(1 LSB) | Factor de escala: x = 1.0 ↔ {cfg.vref:g} V → v = x·Vref; código = round(v/Δ) + 2^(N−1)\n"
            f"SQNR medida = {qr.sqnr_db:.1f} dB | esperada (P_señal/(Δ²/12)) = {qr.sqnr_expected_db:.1f} dB | "
            f"senoide a plena escala = {qr.sqnr_sine_db:.1f} dB | rango dinámico ≈ 6.02·N = {cfg.dynamic_range_db:.1f} dB | "
            f"x(n) a fs = {smp.fs_eff:.0f} Hz{clip}"))
        self.table.delete(*self.table.get_children())
        for row in sample_table(smp.t_n, smp.x_n, qr, i0, 24):
            self.table.insert("", "end", values=row)
