"""Cuantización uniforme, ADC y DAC (apartado d).

Modelo: ADC bipolar de N bits con rango ±Vref (mid-tread).
    Δ (LSB) = 2·Vref / 2^N
    v(n)    = x(n)·Vref                      [V]   (factor de escala = Vref)
    q(n)    = clip(round(v/Δ), -2^(N-1), 2^(N-1)-1)   código con signo
    código ADC (offset binario) = q + 2^(N-1)  ∈ [0, 2^N - 1]
    DAC: v_q(n) = q(n)·Δ
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdcConfig:
    bits: int
    vref: float

    @property
    def levels(self) -> int:
        return 2 ** self.bits

    @property
    def lsb(self) -> float:
        """Resolución (1 LSB) del ADC-DAC en voltios."""
        return 2.0 * self.vref / self.levels

    @property
    def scale_adc(self) -> float:
        """Factor de escala del ADC: niveles (códigos) por voltio."""
        return 1.0 / self.lsb

    @property
    def scale_dac(self) -> float:
        """Factor de escala del DAC: voltios por nivel (LSB)."""
        return self.lsb

    @property
    def vpp(self) -> float:
        """Rango de entrada en voltios (de -Vref a +Vref)."""
        return 2.0 * self.vref

    @property
    def dynamic_range_db(self) -> float:
        return 6.02 * self.bits


@dataclass
class QuantResult:
    cfg: AdcConfig
    volts: np.ndarray        # v(n) antes de cuantizar [V]
    codes: np.ndarray        # q(n) con signo
    codes_u: np.ndarray      # código ADC en offset binario (= índice del nivel)
    volts_q: np.ndarray      # salida del DAC [V]
    error: np.ndarray        # v_q - v [V]
    sqnr_db: float           # medida sobre esta señal
    sqnr_expected_db: float  # esperada para esta señal: P_señal / (Δ²/12)
    sqnr_sine_db: float      # senoide a plena escala: 6.02·N + 1.76
    clipped: int


def _sqnr(p_sig: float, p_err: float) -> float:
    return float(10 * np.log10(p_sig / p_err)) if p_err > 0 and p_sig > 0 else float("inf")


def quantize(x_norm: np.ndarray, cfg: AdcConfig) -> QuantResult:
    v = x_norm * cfg.vref
    half = cfg.levels // 2
    raw = np.round(v / cfg.lsb)
    q = np.clip(raw, -half, half - 1).astype(np.int64)
    vq = q * cfg.lsb
    err = vq - v
    p_sig = float(np.mean(v ** 2))
    return QuantResult(
        cfg=cfg, volts=v, codes=q, codes_u=q + half, volts_q=vq, error=err,
        sqnr_db=_sqnr(p_sig, float(np.mean(err ** 2))),
        sqnr_expected_db=_sqnr(p_sig, cfg.lsb ** 2 / 12.0),
        sqnr_sine_db=6.02 * cfg.bits + 1.76,
        clipped=int(np.sum(raw != q)),
    )


def dac_output(codes_u: np.ndarray, cfg: AdcConfig) -> np.ndarray:
    """DAC ideal: código ADC (offset binario) -> tensión [V]."""
    return (codes_u.astype(np.int64) - cfg.levels // 2) * cfg.lsb


def to_binary(code_u: int, bits: int) -> str:
    return format(int(code_u), f"0{bits}b")


def level_voltages(cfg: AdcConfig) -> np.ndarray:
    """Tensión de cada nivel del ADC, indexada por su código en offset binario.

    V(i) = (i − 2^(N−1))·Δ.  Al ser 2^N par, existe un nivel exacto en 0 V
    (mid-tread); el primero queda en −Vref y el último en +Vref − Δ.
    """
    half = cfg.levels // 2
    return (np.arange(cfg.levels, dtype=np.float64) - half) * cfg.lsb


def nivel_table(cfg: AdcConfig, codes_u: np.ndarray | None = None,
                max_rows: int = 64) -> tuple[list[tuple[str, ...]], str]:
    """Mapeo nivel ↔ voltaje ↔ código binario, y una nota al pie.

    Con `codes_u` se listan solo los códigos que usa realmente la señal (útil
    cuando N es grande y los 2^N niveles no caben en pantalla).
    """
    if codes_u is None:
        idx = np.arange(cfg.levels)
        note = f"Se muestran los {cfg.levels} niveles (2^{cfg.bits})."
    else:
        used = np.unique(np.asarray(codes_u, dtype=np.int64))
        idx = used
        note = (f"Se muestran los {len(used)} códigos usados por la señal "
                f"de {cfg.levels} posibles (2^{cfg.bits}).")
    if len(idx) > max_rows:
        idx = np.concatenate((idx[:max_rows // 2], idx[-(max_rows // 2):]))
        note += f" (se recortan los intermedios: {max_rows} filas de {len(used)})."
    volts = (idx.astype(np.float64) - cfg.levels // 2) * cfg.lsb
    rows = [(str(int(i)), f"{v:+.5f}", to_binary(int(i), cfg.bits))
            for i, v in zip(idx, volts)]
    return rows, note


def pcm_stream(codes_u: np.ndarray, cfg: AdcConfig, count: int = 16) -> str:
    """Trama de bits PCM: los códigos en binario, separados por espacios."""
    n = min(count, len(codes_u))
    return " ".join(to_binary(int(c), cfg.bits) for c in codes_u[:n])


def sqnr_vs_bits(x_norm: np.ndarray, vref: float, bits_list: list[int]) -> list[float]:
    return [quantize(x_norm, AdcConfig(b, vref)).sqnr_db for b in bits_list]


def bits_comparison(x_norm: np.ndarray, vref: float, bits_list: list[int]) -> list[dict]:
    """Tabla Bits | Niveles | Δ (V) | SQNR medida | SQNR teórica."""
    out = []
    for b in bits_list:
        qr = quantize(x_norm, AdcConfig(b, vref))
        out.append({
            "bits": b,
            "levels": qr.cfg.levels,
            "lsb": qr.cfg.lsb,
            "sqnr_db": qr.sqnr_db,
            "sqnr_theo_db": qr.sqnr_sine_db,
        })
    return out


def sample_table(t_n: np.ndarray, x_n: np.ndarray, qr: QuantResult,
                 start: int, rows: int) -> list[tuple[str, ...]]:
    """Filas (n, t, x(n), x(n) en V, q, nivel, código binario, V_DAC, error) para la tabla."""
    end = min(len(x_n), start + rows)
    out = []
    for i in range(start, end):
        out.append((
            str(i),
            f"{t_n[i] * 1e3:.3f}",
            f"{x_n[i]:+.5f}",
            f"{qr.volts[i]:+.5f}",
            f"{qr.codes[i]:+d}",
            str(int(qr.codes_u[i])),
            to_binary(qr.codes_u[i], qr.cfg.bits),
            f"{qr.volts_q[i]:+.5f}",
            f"{qr.error[i] * 1e3:+.3f}",
        ))
    return out
