"""Clases base de las pestañas con figura Matplotlib incrustada."""
from __future__ import annotations

import tkinter as tk
import traceback
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from core.errors import UserError

NO_DATA = "Cargue al menos un audio WAV en la pestaña «0. Carga de audios»."


class PlotTab(ttk.Frame):
    """Pestaña con controles arriba, figura al centro y un área inferior libre."""
    slug = "tab"

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._rendered_version = -1
        self.controls = ttk.Frame(self)
        self.controls.pack(side="top", fill="x", padx=6, pady=4)
        self.bottom = ttk.Frame(self)
        self.bottom.pack(side="bottom", fill="x")
        self.figure = Figure(figsize=(11, 7.5), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    # --- ciclo de vida -------------------------------------------------
    def on_show(self) -> None:
        self.refresh_controls()
        if self._rendered_version != self.app.version:
            self.render()

    def refresh_controls(self) -> None:
        pass

    def render(self) -> None:
        self.figure.clear()
        self.clear_text()
        try:
            self.draw()
        except UserError as exc:
            messagebox.showwarning("Aviso", str(exc))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            messagebox.showerror("Error al graficar", f"{type(exc).__name__}: {exc}")
        self.canvas.draw_idle()
        self._rendered_version = self.app.version

    def clear_text(self) -> None:
        """Vacía los rótulos de texto: se llama antes de cada dibujo."""

    def draw(self) -> None:
        raise NotImplementedError

    def placeholder(self, msg: str = NO_DATA) -> None:
        self.figure.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, color="#777")


class GenreTab(PlotTab):
    """Pestaña que analiza un género a la vez, elegido en un combobox."""

    def __init__(self, master, app):
        super().__init__(master, app)
        ttk.Label(self.controls, text="Género:").pack(side="left")
        self.genre_var = tk.StringVar()
        self.combo = ttk.Combobox(self.controls, textvariable=self.genre_var, state="readonly", width=26)
        self.combo.pack(side="left", padx=4)
        self.combo.bind("<<ComboboxSelected>>", lambda _e: self.render())
        ttk.Button(self.controls, text="Graficar", command=self.render).pack(side="left", padx=4)
        self.build_controls()

    def build_controls(self) -> None:
        pass

    def refresh_controls(self) -> None:
        labels = [t.genre.label for t in self.app.session.loaded()]
        self.combo["values"] = labels
        if self.genre_var.get() not in labels:
            self.genre_var.set(labels[0] if labels else "")

    def current_track(self):
        for t in self.app.session.loaded():
            if t.genre.label == self.genre_var.get():
                return t
        return None

    def draw(self) -> None:
        track = self.current_track()
        if track is None:
            self.placeholder()
            return
        self.draw_track(track, self.app.read_params())

    def draw_track(self, track, params) -> None:
        raise NotImplementedError
