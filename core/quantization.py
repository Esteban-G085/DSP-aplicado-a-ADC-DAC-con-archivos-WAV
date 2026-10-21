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
        return 2.0 * self.vref / self.levels

    @property
    def dynamic_range_db(self) -> float:
        return 6.02 * self.bits


@dataclass
class QuantResult:
    cfg: AdcConfig
    volts: np.ndarray        # v(n) antes de cuantizar [V]
    codes: np.ndarray        # q(n) con signo
    codes_u: np.ndarray      # código ADC en offset binario
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


def sqnr_vs_bits(x_norm: np.ndarray, vref: float, bits_list: list[int]) -> list[float]:
    return [quantize(x_norm, AdcConfig(b, vref)).sqnr_db for b in bits_list]


def sample_table(t_n: np.ndarray, x_n: np.ndarray, qr: QuantResult,
                 start: int, rows: int) -> list[tuple[str, ...]]:
    """Filas (n, t, x(n), x(n) en V, q, código binario, V_DAC, error) para la tabla."""
    end = min(len(x_n), start + rows)
    out = []
    for i in range(start, end):
        out.append((
            str(i),
            f"{t_n[i] * 1e3:.3f}",
            f"{x_n[i]:+.5f}",
            f"{qr.volts[i]:+.5f}",
            f"{qr.codes[i]:+d}",
            to_binary(qr.codes_u[i], qr.cfg.bits),
            f"{qr.volts_q[i]:+.5f}",
            f"{qr.error[i] * 1e3:+.3f}",
        ))
    return out
