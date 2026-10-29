"""Pestaña 2: cuantización, ADC y DAC sobre x(n) (apartado d)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from core.pipeline import extract_segment, sample_segment
from core.quantization import (AdcConfig, bits_comparison, level_voltages, nivel_table, pcm_stream,
                               quantize, sample_table, to_binary)
from core.sampling import pick_zoom_start
from gui.plot_tab import GenreTab
from gui.plot_utils import fmt, fmt_short
from gui.scroll_frame import ScrollFrame

COLUMNS = (("n", 45), ("t [ms]", 75), ("x(n) [norm]", 90), ("x(n) [V]", 90), ("q(n)", 75),
           ("Nivel", 55), ("código ADC (bin)", 130), ("V_DAC [V]", 90), ("error [mV]", 85))
NIVEL_COLUMNS = (("Nivel", 60), ("Voltaje [V]", 100), ("Código binario", 150))

COMPARE_BITS = (2, 3, 4, 5, 6, 7, 8)   # resoluciones comparadas en el panel multi-bits
BOTTOM_HEIGHT = 110                    # alto del panel de texto y tablas (con desplazamiento)
DETAIL_MIN, DETAIL_MAX = 4, 24         # rango del deslizador «muestras en el detalle»


class QuantizationTab(GenreTab):
    slug = "2_cuantizacion_adc_dac"
    bottom_height = BOTTOM_HEIGHT

    def build_controls(self) -> None:
        ttk.Label(self.controls, text="  Muestras en el detalle:").pack(side="left")
        self.n_detail = tk.IntVar(value=12)
        self.n_detail_label = ttk.Label(self.controls, text="12", width=3, anchor="w")
        scale = ttk.Scale(self.controls, from_=DETAIL_MIN, to=DETAIL_MAX, orient="horizontal",
                          variable=self.n_detail, length=170)
        scale.pack(side="left", padx=(4, 2))
        scale.configure(command=self._on_slider)
        scale.bind("<ButtonRelease-1>", lambda _e: self.render())
        self.n_detail_label.pack(side="left")

        # El texto y las tablas van en un panel con desplazamiento: así no
        # aprietan la figura y el usuario les da el espacio que quiera.
        self.scroller = ScrollFrame(self.bottom, height=BOTTOM_HEIGHT)
        self.scroller.pack(fill="both", expand=True)
        body = self.scroller.body

        self.info_label = ttk.Label(body, justify="left", wraplength=1200)
        self.info_label.pack(fill="x", padx=6, pady=(4, 2))

        frame = ttk.Frame(body)
        frame.pack(fill="both", expand=True, padx=6, pady=(0, 4))

        left = ttk.LabelFrame(frame, text="Muestras de x(n) y su código")
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.table = ttk.Treeview(left, columns=[c[0] for c in COLUMNS], show="headings", height=6)
        for name, width in COLUMNS:
            self.table.heading(name, text=name)
            self.table.column(name, width=width, anchor="center")
        sb = ttk.Scrollbar(left, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=sb.set)
        self.table.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        right = ttk.LabelFrame(frame, text="Niveles del ADC ↔ código binario")
        right.pack(side="left", fill="both", expand=True)
        self.nivel_table = ttk.Treeview(right, columns=[c[0] for c in NIVEL_COLUMNS],
                                        show="headings", height=6)
        for name, width in NIVEL_COLUMNS:
            self.nivel_table.heading(name, text=name)
            self.nivel_table.column(name, width=width, anchor="center")
        sb2 = ttk.Scrollbar(right, orient="vertical", command=self.nivel_table.yview)
        self.nivel_table.configure(yscrollcommand=sb2.set)
        self.nivel_table.pack(side="left", fill="both", expand=True)
        sb2.pack(side="right", fill="y")

        self.pcm_label = ttk.Label(body, justify="left", wraplength=1200,
                                   font=("Consolas", 9))
        self.pcm_label.pack(fill="x", padx=6, pady=(0, 6))

    def _on_slider(self, value) -> None:
        self.n_detail_label.config(text=str(int(float(value))))

    def clear_text(self) -> None:
        self.info_label.config(text="")
        self.pcm_label.config(text="")
        self.table.delete(*self.table.get_children())
        self.nivel_table.delete(*self.nivel_table.get_children())

    def draw_track(self, track, params) -> None:
        seg = extract_segment(track.audio, params)
        smp = sample_segment(seg, params)
        cfg = AdcConfig(params.bits, params.vref)
        qr = quantize(smp.x_n, cfg)
        color = track.genre.color
        fig = self.figure
        fig.suptitle(f"{track.title()} — ADC/DAC de {cfg.bits} bits, ±{params.vref:g} V", fontsize=11)
        gs = fig.add_gridspec(2, 2)

        # --- (0,0) x(n) cuantizada y error e(n) (eje gemelo derecho) -----
        win = 0.02
        i0 = pick_zoom_start(smp.x_n, smp.fs_eff, win)
        n = max(8, min(len(smp.x_n) - i0, int(win * smp.fs_eff)))
        sl = slice(i0, i0 + n)
        tm = (smp.t_n[sl] - smp.t_n[i0]) * 1e3

        ax1 = fig.add_subplot(gs[0, 0])
        v, vq = qr.volts[sl], qr.volts_q[sl]
        if cfg.levels <= 16:      # con muchos bits los niveles no se distinguen
            k0 = int(np.floor(v.min() / cfg.lsb))
            k1 = int(np.ceil(v.max() / cfg.lsb))
            for k in range(k0, k1 + 1):
                ax1.axhline(k * cfg.lsb, color="0.88", lw=0.5, zorder=0)
        ax1.plot(tm, v, "o-", color="0.55", ms=3, lw=0.8, label="x(n) [V]")
        ax1.step(tm, vq, where="mid", color=color, lw=1.4, label="salida DAC (cuantizada)")
        ax1.set(xlabel="t [ms]", ylabel="señal [V]")

        ax1e = ax1.twinx()                       # eje gemelo: el error de cuantización
        low_bits = max(2, cfg.bits - 2)
        qr_low = quantize(smp.x_n, AdcConfig(low_bits, cfg.vref))
        ax1e.plot(tm, qr_low.error[sl] * 1e3, color="0.78", lw=0.9,
                  label=f"e(n) con {low_bits} bits (SQNR {qr_low.sqnr_db:.1f} dB)")
        ax1e.plot(tm, qr.error[sl] * 1e3, color="#c0392b", lw=0.9,
                  label=f"e(n) con {cfg.bits} bits (SQNR {qr.sqnr_db:.1f} dB)")
        for s in (+1, -1):
            ax1e.axhline(s * cfg.lsb / 2 * 1e3, color="k", ls="--", lw=0.8)
        ax1e.set_ylabel("error e(n) [mV]")
        ax1.set(title=f"x(n) cuantizada y error e(n) en el eje derecho "
                      f"(Δ = {fmt(cfg.lsb * 1e3, 4)} mV, líneas: ±Δ/2)")
        ax1.grid(alpha=0.3)
        # Las dos leyendas juntas: la de la señal y la del error.
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1e.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                   ncol=2, fontsize=8)

        # --- (0,1) comparación de resoluciones ----------------------------
        bits_list = sorted(set(COMPARE_BITS) | {cfg.bits})
        ax3 = fig.add_subplot(gs[0, 1])
        ax3.plot(tm, qr.volts[sl], "o", color="0.6", ms=3, label="x(n) [V]")
        shades = ("0.55", "0.45", "0.35", "#1e8449", "#2471a3", "#7b5ea7", "#d68910")
        for i, b in enumerate(bits_list):
            qb = quantize(smp.x_n, AdcConfig(b, cfg.vref))
            ax3.step(tm, qb.volts_q[sl], where="mid", lw=1.2,
                     color=(color if b == cfg.bits else shades[i % len(shades)]),
                     ls=("-" if b == cfg.bits else "--"),
                     label=f"{b} b (Δ={fmt_short(qb.cfg.lsb * 1e3, 3)} mV)")
        ax3.set(title=f"La misma x(n) cuantizada con distintas resoluciones "
                      f"({bits_list[0]} a {bits_list[-1]} bits, L = 2ᴺ niveles)",
                xlabel="t [ms]", ylabel="V")
        ax3.grid(alpha=0.3)
        ax3.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=4)

        # --- (1,0) SQNR frente al número de bits --------------------------
        ax4 = fig.add_subplot(gs[1, 0])
        bl = list(range(2, 25))
        ax4.plot(bl, [quantize(smp.x_n, AdcConfig(b, cfg.vref)).sqnr_db for b in bl], "o-",
                 ms=3, color=color, label="medida en esta señal")
        ax4.plot(bl, [6.02 * b + 1.76 for b in bl], "k--", lw=1,
                 label="senoide a plena escala: 6.02·N + 1.76")
        ax4.axvline(cfg.bits, color="#c0392b", lw=1.2)
        ax4.plot([cfg.bits], [qr.sqnr_db], "s", color="#c0392b",
                 label=f"N = {cfg.bits}: {qr.sqnr_db:.1f} dB")
        ax4.set(title="Relación señal-ruido de cuantización (SQNR) vs bits", xlabel="N [bits]",
                ylabel="SQNR [dB]")
        ax4.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
        ax4.grid(alpha=0.3)

        # --- (1,1) detalle con los códigos binarios -----------------------
        n_det = max(DETAIL_MIN, min(DETAIL_MAX, int(self.n_detail.get())))
        j0 = pick_zoom_start(smp.x_n, smp.fs_eff, n_det / smp.fs_eff)
        k = min(n_det, len(smp.x_n) - j0)
        sd = slice(j0, j0 + k)
        td = (smp.t_n[sd] - smp.t_n[j0]) * 1e3

        ax6 = fig.add_subplot(gs[1, 1])
        # Con muchos bits los 2^N niveles no se pueden dibujar: se muestran solo si son pocos.
        few_levels = cfg.levels <= 32
        levels = level_voltages(cfg) if few_levels else None
        if few_levels:
            for lv in levels:
                ax6.axhline(lv, color="0.85", ls=":", lw=0.6)
        ax6.step(td, qr.volts_q[sd], where="mid", color=color, lw=1.4,
                 label=f"x_q(n) — {cfg.bits} bits")
        markers, stems, _ = ax6.stem(td, qr.volts[sd], basefmt=" ")
        markers.set(color="0.45", markersize=4)
        stems.set(color="0.45", linewidth=0.8)

        # Anotar todos los códigos a la vez los solapaba: se salta alguno de cada
        # grupo según lo largo que sea la etiqueta frente al sitio disponible,
        # y se alterna la altura para que las que sí se dibujan no se pisen.
        panel_w = 0.46 * fig.get_figwidth() * fig.dpi          # ancho aprox. del panel
        spacing = panel_w / max(1, k)
        label_w = cfg.bits * 4.8                              # ancho aprox. del texto
        stride = max(1, int(np.ceil(label_w / max(spacing, 1.0))))
        for i, (t, val, code) in enumerate(zip(td, qr.volts_q[sd], qr.codes_u[sd])):
            if i % stride:
                continue
            dy = 9 if (i // stride) % 2 == 0 else -17
            ax6.annotate(to_binary(int(code), cfg.bits), (t, val),
                         textcoords="offset points", xytext=(0, dy), ha="center",
                         fontsize=7, fontweight="bold", color="#7b241c")
        ax6.margins(y=0.30)
        if few_levels:
            ax6.set_yticks(levels)
            ax6.set_yticklabels([f"{i:0{cfg.bits}b} ({v:+.2f}V)" for i, v in enumerate(levels)],
                                fontfamily="monospace", fontsize=7)
        ax6.set(title=f"Codificación binaria PCM (L={cfg.levels} niveles, "
                       f"Δ = {fmt(cfg.lsb)} V)", xlabel="t [ms]", ylabel="código binario (V)")
        ax6.grid(alpha=0.3, which="both")
        ax6.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2)

        # --- bloques de texto ---------------------------------------------
        clip = ""
        if qr.clipped:
            plural = "" if qr.clipped == 1 else "s"
            clip = f" | {qr.clipped} muestra{plural} saturada{plural}"
        used_lo, used_hi = int(qr.codes_u.min()), int(qr.codes_u.max())
        uso = 100.0 * (used_hi - used_lo + 1) / max(1, cfg.levels)
        self.info_label.config(text=(
            f"Resolución: {cfg.bits} bits ({cfg.levels} niveles) | "
            f"Rango de entrada: [−{params.vref:g} V, +{params.vref:g} V] "
            f"(Vpp = {cfg.vpp:.2f} V) | "
            f"Tamaño del paso / LSB: Δ = 2·Vref/2^N = {fmt(cfg.lsb * 1e3, 4)} mV "
            f"({fmt(cfg.lsb)} V) | Factor de escala ADC: {fmt(cfg.scale_adc, 3)} Niveles/Voltio | "
            f"Factor de escala DAC: {fmt(cfg.scale_dac)} Voltios/Nivel\n"
            f"SQNR medida = {qr.sqnr_db:.1f} dB | esperada (P_señal/(Δ²/12)) = "
            f"{qr.sqnr_expected_db:.1f} dB | senoide a plena escala = {qr.sqnr_sine_db:.1f} dB | "
            f"rango dinámico ≈ 6.02·N = {cfg.dynamic_range_db:.1f} dB | "
            f"Uso del rango del ADC: códigos {used_lo}…{used_hi} de {cfg.levels - 1} "
            f"({uso:.0f} % del rango) | x(n) a fs = {smp.fs_eff:.0f} Hz{clip}"))

        self.table.delete(*self.table.get_children())
        for row in sample_table(smp.t_n, smp.x_n, qr, j0, 40):
            self.table.insert("", "end", values=row)

        rows, note = nivel_table(cfg, qr.codes_u)
        self.nivel_table.delete(*self.nivel_table.get_children())
        for row in rows:
            self.nivel_table.insert("", "end", values=row)

        comp = bits_comparison(smp.x_n, cfg.vref, bits_list)
        head = "  ".join(f"{b:>2d} bits" for b in bits_list)
        lines = [f"Bits:              {head}"]
        lines.append("Niveles (L):       " + "  ".join(f"{c['levels']:>8d}" for c in comp))
        lines.append("Δ (V):             " + "  ".join(f"{c['lsb']:>8.4f}" for c in comp))
        lines.append("SQNR medida (dB):  " + "  ".join(f"{c['sqnr_db']:>8.2f}" for c in comp))
        lines.append("SQNR teórica (dB): " + "  ".join(f"{c['sqnr_theo_db']:>8.2f}" for c in comp))
        lines.append("")
        lines.append(f"Trama PCM (primeros {min(16, len(smp.x_n))} códigos, {cfg.bits} bits):")
        lines.append(pcm_stream(qr.codes_u, cfg, 16))
        lines.append("")
        lines.append(note)
        self.pcm_label.config(text="\n".join(lines))
