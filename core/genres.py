"""Definición de los cinco géneros musicales exigidos por la guía."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenreDef:
    key: str
    label: str
    color: str
    hint: str


GENRES: tuple[GenreDef, ...] = (
    GenreDef("barroca", "Clásica - Barroca", "#7b5ea7",
             "p. ej. Bach, Vivaldi, Händel — orquesta / sinfónica / filarmónica"),
    GenreDef("metal", "Metal", "#c0392b",
             "banda de metal (indique banda, álbum y año)"),
    GenreDef("salsa_electronica", "Salsa o Electrónica", "#d68910",
             "orquesta de salsa o artista de música electrónica"),
    GenreDef("cantina", "Cantina al hombro", "#1e8449",
             "intérprete / agrupación y canción"),
    GenreDef("reggaeton", "Reggaeton", "#2471a3",
             "artista y canción"),
)

GENRE_BY_KEY = {g.key: g for g in GENRES}
