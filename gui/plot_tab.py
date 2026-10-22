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
    """Pestaña con controles arriba, figura al centro y un panel inferior
    separado por una barra arrastrable: el usuario decide cuánto espacio
    dan a las gráficas y cuánto al texto y las tablas, y puede plegar el
    panel con el botón «Solo gráficas»."""
    slug = "tab"
    bottom_height = 150          # alto del panel inferior cuando se despliega [px]

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._rendered_version = -1
        self._bottom_hidden = False

        self.controls = ttk.Frame(self)
        self.controls.pack(side="top", fill="x", padx=6, pady=4)

        self.pane = ttk.Panedwindow(self, orient="vertical")
        self.pane.pack(side="top", fill="both", expand=True)

        # panel superior: barra de herramientas + figura
        self.plot_area = ttk.Frame(self.pane)
        # La figura pide su tamaño en pulgadas (750 px), y eso desbordaría el
        # panel: se declara un alto mínimo para que el reparto lo haga la
        # barra divisoria y no el tamaño solicitado de la figura.
        self.plot_area.configure(height=150)
        self.plot_area.pack_propagate(False)
        self.figure = Figure(figsize=(11, 7.5), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_area)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_area, pack_toolbar=False)
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # panel inferior: texto, tablas y trama PCM (con desplazamiento)
        self.bottom = ttk.Frame(self.pane)
        self.pane.add(self.plot_area, weight=1)
        self.pane.add(self.bottom, weight=0)

        self.collapse_btn = ttk.Button(self.controls, text="Solo gráficas",
                                       command=self.toggle_bottom)
        self.collapse_btn.pack(side="right")

    # --- reparto del espacio -------------------------------------------
    def set_bottom_height(self, px: int) -> None:
        """Fija el alto del panel inferior; la figura toma todo el resto."""
        total = self.pane.winfo_height()
        if total > 1:
            self.pane.sashpos(0, max(0, total - px))

    def toggle_bottom(self) -> None:
        """Plega o despliega el panel de texto y tablas."""
        self._bottom_hidden = not self._bottom_hidden
        self.set_bottom_height(0 if self._bottom_hidden else self.bottom_height)
        self.collapse_btn.config(text="Mostrar detalle" if self._bottom_hidden else "Solo gráficas")

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
