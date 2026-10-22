"""Pestaña 0: carga de los cinco WAV e investigación de cada obra (apartados a y b)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from core.genres import GENRES
from gui.genre_slot import GenreSlot


class LoadTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        ttk.Label(self, wraplength=1200, justify="left", text=(
            "Cargue un WAV corto por género y complete la investigación breve (músico/orquesta/banda, obra y año). "
            "Estéreo se convierte a mono promediando canales. Luego use las pestañas 1 a 3.")
        ).pack(fill="x", padx=10, pady=(8, 4))
        self.slots: dict[str, GenreSlot] = {}
        for g in GENRES:
            slot = GenreSlot(self, app, g)
            slot.pack(fill="x", padx=10, pady=4)
            self.slots[g.key] = slot

    def refresh_controls(self) -> None:
        for s in self.slots.values():
            s.refresh()

    def on_show(self) -> None:
        self.refresh_controls()
