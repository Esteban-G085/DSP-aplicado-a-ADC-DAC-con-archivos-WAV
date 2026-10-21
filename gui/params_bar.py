"""Barra de parámetros globales (segmento, muestreo, bits y Vref)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from core.errors import ParamsError
from core.pipeline import Params


class ParamsBar(ttk.LabelFrame):
    def __init__(self, master, on_change: Callable[[], None]):
        super().__init__(master, text="Parámetros globales (aplican a todas las pestañas)")
        d = Params()
        self.v = {
            "start": tk.StringVar(value=str(d.start_s)),
            "dur": tk.StringVar(value=str(d.duration_s)),
            "fs": tk.StringVar(value=str(d.fs_target)),
            "bits": tk.StringVar(value=str(d.bits)),
            "vref": tk.StringVar(value=str(d.vref)),
            "aa": tk.BooleanVar(value=d.antialias),
            "norm": tk.BooleanVar(value=d.normalize),
        }
        for var in self.v.values():
            var.trace_add("write", lambda *_: on_change())

        col = 0

        def field(label, key, width, values=None):
            nonlocal col
            ttk.Label(self, text=label).grid(row=0, column=col, padx=(8, 2), pady=4, sticky="e")
            if values:
                w = ttk.Combobox(self, textvariable=self.v[key], values=values, width=width)
            else:
                w = ttk.Entry(self, textvariable=self.v[key], width=width)
            w.grid(row=0, column=col + 1, pady=4)
            col += 2

        field("Inicio [s]", "start", 6)
        field("Duración [s]", "dur", 6)
        field("fs de x(n) [Hz]", "fs", 8, ["44100", "22050", "11025", "8000", "4000", "2000"])
        field("Bits ADC/DAC (N)", "bits", 4, ["4", "6", "8", "10", "12", "16", "24"])
        field("±Vref [V]", "vref", 6, ["1.0", "1.65", "2.5", "3.3", "5.0"])
        ttk.Checkbutton(self, text="Filtro antialiasing", variable=self.v["aa"]).grid(row=0, column=col, padx=8)
        ttk.Checkbutton(self, text="Normalizar pico", variable=self.v["norm"]).grid(row=0, column=col + 1, padx=8)

    def get(self) -> Params:
        try:
            p = Params(
                start_s=float(self.v["start"].get().replace(",", ".")),
                duration_s=float(self.v["dur"].get().replace(",", ".")),
                fs_target=int(float(self.v["fs"].get().replace(",", "."))),
                antialias=bool(self.v["aa"].get()),
                bits=int(float(self.v["bits"].get())),
                vref=float(self.v["vref"].get().replace(",", ".")),
                normalize=bool(self.v["norm"].get()),
            )
        except ValueError as exc:
            raise ParamsError("Revise los parámetros globales: hay un valor que no es un número.") from exc
        return p.validate()
