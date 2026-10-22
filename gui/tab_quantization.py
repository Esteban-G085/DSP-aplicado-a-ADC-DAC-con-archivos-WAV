"""Pestaña 2: cuantización, ADC y DAC sobre x(n) (apartado d)."""
from __future__ import annotations

import numpy as np

from core.pipeline import extract_segment, sample_segment
from core.quantization import AdcConfig, quantize
from core.sampling import pick_zoom_start
from gui.plot_tab import GenreTab
from gui.plot_utils import fmt

COMPARE_BITS = (2, 3, 4, 5, 6, 7, 8)


class QuantizationTab(GenreTab):
    slug = "2_cuantizacion_adc_dac"

    VIEWS = {"time", "resolutions"}

    def __init__(self, master, app, view: str):
        if view not in self.VIEWS:
            raise ValueError(f"Vista de cuantización desconocida: {view}")
        self.view = view
        super().__init__(master, app)

    def draw_track(self, track, params) -> None:
        seg = extract_segment(track.audio, params)
        smp = sample_segment(seg, params)
        cfg = AdcConfig(params.bits, params.vref)
        qr = quantize(smp.x_n, cfg)
        color = track.genre.color
        fig = self.figure
        fig.suptitle(f"{track.title()} — ADC/DAC de {cfg.bits} bits, ±{params.vref:g} V", fontsize=11)
        gs = fig.add_gridspec(1, 2) if self.view == "time" else fig.add_gridspec(1, 1)

        win = 0.02
        i0 = pick_zoom_start(smp.x_n, smp.fs_eff, win)
        n = max(8, min(len(smp.x_n) - i0, int(win * smp.fs_eff)))
        sl = slice(i0, i0 + n)
        tm = (smp.t_n[sl] - smp.t_n[i0]) * 1e3

        if self.view == "time":
            ax1 = fig.add_subplot(gs[0, 0])
            v, vq = qr.volts[sl], qr.volts_q[sl]
            if cfg.levels <= 16:
                k0 = int(np.floor(v.min() / cfg.lsb))
                k1 = int(np.ceil(v.max() / cfg.lsb))
                for k in range(k0, k1 + 1):
                    ax1.axhline(k * cfg.lsb, color="0.88", lw=0.5, zorder=0)
            ax1.plot(tm, v, "o-", color="0.55", ms=3, lw=0.8, label="x(n) [V]")
            ax1.step(tm, vq, where="mid", color=color, lw=1.4, label="salida DAC (cuantizada)")
            ax1.set(title=f"x(n) y cuantización (Δ = {fmt(cfg.lsb * 1e3, 4)} mV)",
                    xlabel="t [ms]", ylabel="V")
            ax1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)

            ax2 = fig.add_subplot(gs[0, 1])
            low_bits = max(2, cfg.bits - 2)
            qr_low = quantize(smp.x_n, AdcConfig(low_bits, cfg.vref))
            ax2.plot(tm, qr_low.error[sl] * 1e3, color="0.8", lw=0.9,
                     label=f"e(n) con {low_bits} bits (SQNR {qr_low.sqnr_db:.1f} dB)")
            ax2.plot(tm, qr.error[sl] * 1e3, color="#c0392b", lw=0.9,
                     label=f"e(n) con {cfg.bits} bits (SQNR {qr.sqnr_db:.1f} dB)")
            for s in (+1, -1):
                ax2.axhline(s * cfg.lsb / 2 * 1e3, color="k", ls="--", lw=0.8)
            ax2.set(title="Error de cuantización e(n) = V_DAC − x(n)   (líneas: ±Δ/2)",
                    xlabel="t [ms]", ylabel="mV")
            ax2.grid(alpha=0.3)
            ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)

        if self.view == "resolutions":
            ax3 = fig.add_subplot(gs[0, 0])
            bits_list = sorted(set(COMPARE_BITS) | {cfg.bits})
            ax3.plot(tm, qr.volts[sl], "o", color="0.6", ms=3, label="x(n) [V]")
            shades = ("0.55", "0.45", "0.35", "#1e8449", "#2471a3", "#7b5ea7", "#d68910")
            for i, bits in enumerate(bits_list):
                comparison = quantize(smp.x_n, AdcConfig(bits, cfg.vref))
                ax3.step(
                    tm,
                    comparison.volts_q[sl],
                    where="mid",
                    lw=1.2,
                    color=color if bits == cfg.bits else shades[i % len(shades)],
                    ls="-" if bits == cfg.bits else "--",
                    label=f"{bits} b (L={2 ** bits}, Δ={fmt(comparison.cfg.lsb * 1e3, 3)} mV)",
                )
            ax3.set(
                title=f"x(n) cuantizada con distintas resoluciones "
                      f"({bits_list[0]} a {bits_list[-1]} bits)",
                xlabel="t [ms]",
                ylabel="V",
            )
            ax3.grid(alpha=0.3)
            ax3.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
