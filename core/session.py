"""Estado de la aplicación: audios cargados y datos de investigación por género."""
from __future__ import annotations

from dataclasses import dataclass

from .audio_io import Audio
from .genres import GENRES, GenreDef


@dataclass
class TrackMeta:
    artist: str = ""   # músico / orquesta / banda
    work: str = ""     # obra o canción
    year: str = ""
    notes: str = ""    # investigación breve


@dataclass
class Track:
    genre: GenreDef
    audio: Audio
    meta: TrackMeta

    @property
    def key(self) -> str:
        return self.genre.key

    def title(self) -> str:
        parts = [self.genre.label]
        detail = " — ".join(p for p in (self.meta.artist, self.meta.work) if p)
        if detail:
            parts.append(detail)
        if self.meta.year:
            parts[-1] += f" ({self.meta.year})"
        return " | ".join(parts)


class Session:
    def __init__(self) -> None:
        self.meta: dict[str, TrackMeta] = {g.key: TrackMeta() for g in GENRES}
        self._audio: dict[str, Audio] = {}

    def set_audio(self, key: str, audio: Audio) -> None:
        self._audio[key] = audio

    def remove(self, key: str) -> None:
        self._audio.pop(key, None)

    def audio(self, key: str) -> Audio | None:
        return self._audio.get(key)

    def loaded(self) -> list[Track]:
        """Pistas cargadas, en el orden fijo de GENRES."""
        return [Track(g, self._audio[g.key], self.meta[g.key])
                for g in GENRES if g.key in self._audio]
