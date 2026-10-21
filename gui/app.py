"""Ventana principal: ensambla pestañas, parámetros globales y menús."""
from __future__ import annotations

import os
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from core.errors import UserError
from core.pipeline import Params
from core.report import analyze_tracks, export_csv
from core.session import Session
from core.synth import write_demo_wavs
from gui.params_bar import ParamsBar
from gui.tab_comparison import ComparisonTab
from gui.tab_integrated import IntegratedTab
from gui.tab_load import LoadTab
from gui.tab_quantization import QuantizationTab
from gui.tab_sampling import SamplingTab
from gui.tab_signal import SignalTab

HELP = (
    "Flujo de trabajo:\n"
    "0. Cargue un WAV por género y escriba la investigación (músico, obra, año).\n"
    "1. x(t): señal en el tiempo, zoom y espectrograma.\n"
    "2. Muestreo y FFT: x(n) a la fs elegida, espectro de magnitud y verificación de Nyquist-Shannon.\n"
    "3. Cuantización: ADC/DAC de N bits y ±Vref, resolución (LSB), SQNR y tabla con los códigos binarios de x(n).\n"
    "4. Comparación: espectros de todos los géneros en una gráfica y ranking TDAH.\n"
    "5. Integrado: todas las señales x(t) en una figura.\n\n"
    "Cambie los parámetros globales (arriba) y vuelva a graficar para ver aliasing (fs baja sin filtro) "
    "o ruido de cuantización (N bajo)."
)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DSP aplicado a ADC–DAC con archivos WAV")
        self.geometry("1320x880")
        self.minsize(1100, 700)
        self.session = Session()
        self.version = 0
        self._tmp_demo: str | None = None

        self.params_bar = ParamsBar(self, on_change=self.bump)
        self.params_bar.pack(fill="x", padx=8, pady=(6, 2))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)
        self.load_tab = LoadTab(self.notebook, self)
        self.plot_tabs = [SignalTab(self.notebook, self), SamplingTab(self.notebook, self),
                          QuantizationTab(self.notebook, self), ComparisonTab(self.notebook, self),
                          IntegratedTab(self.notebook, self)]
        self.tabs = [self.load_tab, *self.plot_tabs]
        titles = ["0. Carga de audios", "1. Señal x(t)", "2. Muestreo y FFT", "3. Cuantización ADC–DAC",
                  "4. Comparación (TDAH)", "5. Integrado x(t)"]
        for tab, title in zip(self.tabs, titles):
            self.notebook.add(tab, text=title)
        self.notebook.bind("<<NotebookTabChanged>>", lambda _e: self.current_tab().on_show())
        self._build_menu()

    # --- estado compartido ---------------------------------------------
    def bump(self) -> None:
        """Invalida las figuras: se redibujan al abrir la pestaña o al pulsar «Graficar»."""
        self.version += 1

    def read_params(self) -> Params:
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
        m.add_command(label="Cargar demo sintética (para probar)", command=self.load_demo)
        m.add_separator()
        m.add_command(label="Exportar figuras (PNG)…", command=self.export_figures)
        m.add_command(label="Exportar métricas (CSV)…", command=self.export_metrics)
        m.add_separator()
        m.add_command(label="Salir", command=self.destroy)
        bar.add_cascade(label="Archivo", menu=m)
        h = tk.Menu(bar, tearoff=False)
        h.add_command(label="Guía de uso", command=lambda: messagebox.showinfo("Guía de uso", HELP))
        bar.add_cascade(label="Ayuda", menu=h)
        self.config(menu=bar)

    def load_demo(self) -> None:
        if self._tmp_demo is None:
            self._tmp_demo = tempfile.mkdtemp(prefix="dsp_demo_")
        try:
            for key, path in write_demo_wavs(self._tmp_demo).items():
                slot = self.load_tab.slots[key]
                slot.set_meta("Sintético (demo)", "señal generada por código", "2026",
                              "Audio de prueba, no es música real: reemplácelo por su WAV.")
                slot.load_path(path)
        except UserError as exc:
            messagebox.showwarning("Demo", str(exc))

    def export_figures(self) -> None:
        if not self.session.loaded():
            messagebox.showinfo("Exportar", "Cargue al menos un audio primero.")
            return
        folder = filedialog.askdirectory(title="Carpeta de destino para las figuras")
        if not folder:
            return
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            saved = [p for tab in self.plot_tabs for p in tab.export(folder)]
        finally:
            self.config(cursor="")
        messagebox.showinfo("Exportar", f"{len(saved)} figuras guardadas en:\n{folder}")

    def export_metrics(self) -> None:
        tracks = self.session.loaded()
        if not tracks:
            messagebox.showinfo("Exportar", "Cargue al menos un audio primero.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="metricas_generos.csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            metrics, ranking = analyze_tracks(tracks, self.read_params())
            export_csv(path, tracks, metrics, ranking)
        except UserError as exc:
            messagebox.showwarning("Exportar", str(exc))
            return
        messagebox.showinfo("Exportar", f"Métricas guardadas en:\n{os.path.abspath(path)}")
