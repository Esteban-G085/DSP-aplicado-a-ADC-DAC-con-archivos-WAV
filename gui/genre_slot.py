"""Ranura de un género: carga del WAV + datos de investigación (apartado b)."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from core.audio_io import load_wav
from core.errors import UserError
from core.genres import GenreDef


class GenreSlot(ttk.LabelFrame):
    def __init__(self, master, app, genre: GenreDef):
        super().__init__(master, text=f"{genre.label}   ({genre.hint})")
        self.app, self.genre = app, genre
        meta = app.session.meta[genre.key]

        top = ttk.Frame(self)
        top.pack(fill="x", padx=6, pady=(4, 2))
        ttk.Button(top, text="Cargar WAV…", command=self.load).pack(side="left")
        ttk.Button(top, text="Quitar", command=self.clear).pack(side="left", padx=4)
        self.info = ttk.Label(top, text="(sin archivo)", foreground="#666")
        self.info.pack(side="left", padx=8)

        row = ttk.Frame(self)
        row.pack(fill="x", padx=6, pady=2)
        self.vars = {}
        for label, attr, width in (("Músico / orquesta / banda:", "artist", 30),
                                   ("Obra / canción:", "work", 30), ("Año:", "year", 6)):
            ttk.Label(row, text=label).pack(side="left", padx=(0, 2))
            var = tk.StringVar(value=getattr(meta, attr))
            var.trace_add("write", lambda *_, a=attr, v=var: self._set_meta(a, v))
            ttk.Entry(row, textvariable=var, width=width).pack(side="left", padx=(0, 10))
            self.vars[attr] = var

        row2 = ttk.Frame(self)
        row2.pack(fill="x", padx=6, pady=(2, 6))
        ttk.Label(row2, text="Investigación breve:").pack(side="left", padx=(0, 2))
        var = tk.StringVar(value=meta.notes)
        var.trace_add("write", lambda *_: self._set_meta("notes", var))
        ttk.Entry(row2, textvariable=var).pack(side="left", fill="x", expand=True)
        self.vars["notes"] = var

    def _set_meta(self, attr: str, var: tk.StringVar) -> None:
        setattr(self.app.session.meta[self.genre.key], attr, var.get())
        self.app.bump()

    def set_meta(self, artist="", work="", year="", notes="") -> None:
        for attr, val in (("artist", artist), ("work", work), ("year", year), ("notes", notes)):
            self.vars[attr].set(val)

    def load_path(self, path: str) -> None:
        audio = load_wav(path)
        self.app.session.set_audio(self.genre.key, audio)
        self.refresh()
        self.app.on_data_changed()

    def load(self) -> None:
        path = filedialog.askopenfilename(title=f"WAV de {self.genre.label}",
                                          filetypes=[("Audio WAV", "*.wav *.WAV"), ("Todos", "*.*")])
        if not path:
            return
        try:
            self.load_path(path)
        except UserError as exc:
            messagebox.showwarning("No se pudo cargar", str(exc))

    def clear(self) -> None:
        self.app.session.remove(self.genre.key)
        self.refresh()
        self.app.on_data_changed()

    def refresh(self) -> None:
        a = self.app.session.audio(self.genre.key)
        if a is None:
            self.info.config(text="(sin archivo)", foreground="#666")
        else:
            kind = "float" if a.is_float else "PCM"
            self.info.config(foreground="#1a5e2a", text=(
                f"{a.name} — fs = {a.fs} Hz, {a.bits} bits {kind}, "
                f"{'estéreo→mono' if a.channels > 1 else 'mono'}, {a.duration:.1f} s"))
