"""Marco con desplazamiento vertical, para que el contenido largo no apriete la figura."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def _theme_bg() -> str:
    """Color de fondo del tema, para que el lienzo no se note."""
    try:
        return ttk.Style().lookup("TFrame", "background") or "SystemButtonFace"
    except Exception:  # noqa: BLE001
        return "SystemButtonFace"


class ScrollFrame(tk.Frame):
    """Contenedor con barra de desplazamiento vertical y alto fijo.

    El alto lo fija quien lo crea, de modo que unas tablas largas o un texto
    de varias líneas no aprietan la figura de Matplotlib que está arriba:
    el espacio sobrante se recorre con la barra.
    """

    def __init__(self, master, height: int = 250):
        bg = _theme_bg()
        super().__init__(master, bg=bg, height=height)
        self.body = tk.Frame(self, bg=bg)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg=bg)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")
        self.body.bind("<Configure>", self._on_body)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.pack_propagate(False)   # el alto no depende del contenido

    def _on_body(self, _event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event) -> None:
        self.canvas.itemconfigure(self._win, width=event.width)

    def _on_wheel(self, event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 60), "units")
