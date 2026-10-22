"""Ventana principal: ensambla pestañas, parámetros globales y menús."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from core.session import Session
from gui.params_bar import ParamsBar
from gui.tab_comparison import ComparisonTab
from gui.tab_load import LoadTab
from gui.tab_quantization import QuantizationTab
from gui.tab_sampling import SamplingTab

HELP = (
    "Flujo de trabajo:\n"
    "0. Cargue un WAV corto por género y escriba la investigación (músico, obra, año).\n"
    "1. Señal, muestreo y FFT: x(t), x(n) con sus bits, espectro de magnitud y Nyquist-Shannon.\n"
    "2. Cuantización ADC–DAC: factor de escala, resolución (LSB), SQNR, códigos binarios y trama PCM.\n"
    "3. Comparación: espectros de magnitud de todos los géneros en una sola gráfica.\n\n"
    "Cambie los parámetros globales (arriba) y vuelva a graficar para ver aliasing (fs baja sin filtro) "
    "o ruido de cuantización (N bajo)."
)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DSP aplicado a ADC–DAC con archivos WAV")
        self.geometry("1400x980")
        self.minsize(1100, 700)
        self.session = Session()
        self.version = 0

        self.params_bar = ParamsBar(self, on_change=self.bump)
        self.params_bar.pack(fill="x", padx=8, pady=(6, 2))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)
        self.load_tab = LoadTab(self.notebook, self)
        self.plot_tabs = [SamplingTab(self.notebook, self), QuantizationTab(self.notebook, self),
                          ComparisonTab(self.notebook, self)]
        self.tabs = [self.load_tab, *self.plot_tabs]
        titles = ["0. Carga de audios", "1. Señal, muestreo y FFT", "2. Cuantización ADC–DAC",
                  "3. Comparación de espectros"]
        for tab, title in zip(self.tabs, titles):
            self.notebook.add(tab, text=title)
        self.notebook.bind("<<NotebookTabChanged>>", lambda _e: self.current_tab().on_show())
        self._build_menu()

    # --- estado compartido ---------------------------------------------
    def bump(self) -> None:
        """Invalida las figuras: se redibujan al abrir la pestaña o al pulsar «Graficar»."""
        self.version += 1

    def read_params(self):
        return self.params_bar.get()

    def on_data_changed(self) -> None:
        self.bump()
        for t in self.tabs:
            t.refresh_controls()

    def current_tab(self):
        return self.tabs[self.notebook.index(self.notebook.select())]

    # --- menús ---------------------------------------------------------
    def _build_menu(self) -> None:
        bar = tk.Menu(self)
        m = tk.Menu(bar, tearoff=False)
        m.add_command(label="Salir", command=self.destroy)
        bar.add_cascade(label="Archivo", menu=m)
        h = tk.Menu(bar, tearoff=False)
        h.add_command(label="Guía de uso", command=lambda: messagebox.showinfo("Guía de uso", HELP))
        bar.add_cascade(label="Ayuda", menu=h)
        self.config(menu=bar)
