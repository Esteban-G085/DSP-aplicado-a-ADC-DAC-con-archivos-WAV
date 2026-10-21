"""Utilidades de graficación compartidas por las pestañas."""
from __future__ import annotations

import numpy as np


def to_db(mag: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(mag + 1e-12)


def plot_decimate(f: np.ndarray, y: np.ndarray, n_out: int = 3000):
    """Reduce un espectro largo a ~n_out puntos conservando los máximos locales."""
    n = len(y)
    if n <= 2 * n_out:
        return f, y
    edges = np.linspace(0, n, n_out + 1).astype(int)[:-1]
    counts = np.diff(np.append(edges, n))
    return np.add.reduceat(f, edges) / counts, np.maximum.reduceat(y, edges)
