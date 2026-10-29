"""Pruebas del núcleo (sin GUI):  python -m unittest discover -s tests"""
import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.audio_io import load_wav  # noqa: E402
from core.errors import AudioError  # noqa: E402
from core.pipeline import Params, extract_segment, sample_segment  # noqa: E402
from core.quantization import (AdcConfig, bits_comparison, dac_output, level_voltages, nivel_table,  # noqa: E402
                               pcm_stream, quantize, to_binary)
from core.sampling import (aliasing_residual, decimate, decimation_step, dominant_frequency,  # noqa: E402
                           effective_bandwidth, lowpass_antialias, magnitude_spectrum, nyquist_report,
                           pick_zoom_start, spectral_centroid)

FS = 44100


def sine(f, a=1.0, dur=2.0, fs=FS):
    t = np.arange(int(dur * fs)) / fs
    return a * np.sin(2 * np.pi * f * t)


class TestSampling(unittest.TestCase):
    def test_peak_frequency_and_amplitude(self):
        f, mag = magnitude_spectrum(sine(1000, 0.5), FS)
        self.assertAlmostEqual(f[np.argmax(mag)], 1000, delta=1)
        self.assertAlmostEqual(mag.max(), 0.5, delta=0.02)

    def test_two_tone_spectrum_preserves_relative_amplitudes(self):
        fs = 1000
        t = np.arange(100) / fs
        x = 3 * np.sin(2 * np.pi * 10 * t) + np.cos(2 * np.pi * 30 * t)
        f, mag = magnitude_spectrum(x, fs)
        self.assertAlmostEqual(mag[np.argmin(abs(f - 10))], 3.0, places=10)
        self.assertAlmostEqual(mag[np.argmin(abs(f - 30))], 1.0, places=10)

    def test_spectrum_does_not_double_dc_or_nyquist(self):
        f_dc, mag_dc = magnitude_spectrum(np.ones(8), 8)
        self.assertAlmostEqual(f_dc[0], 0)
        self.assertAlmostEqual(mag_dc[0], 1)

        f_nyquist, mag_nyquist = magnitude_spectrum(np.array([1, -1] * 4), 8)
        self.assertAlmostEqual(f_nyquist[-1], 4)
        self.assertAlmostEqual(mag_nyquist[-1], 1)

    def test_bandwidth(self):
        f, mag = magnitude_spectrum(sine(3000), FS)
        self.assertLess(abs(effective_bandwidth(f, mag) - 3000), 30)

    def test_aliasing_without_filter(self):
        # 5 kHz muestreada a 8 kHz sin antialiasing -> aparece en |5000-8000| = 3000 Hz
        step, fs_eff, xn = decimate(sine(5000, fs=48000), 48000, 8000)
        f, mag = magnitude_spectrum(xn, fs_eff)
        self.assertEqual(step, 6)
        self.assertAlmostEqual(f[np.argmax(mag)], 3000, delta=5)

    def test_antialias_removes_it(self):
        x = lowpass_antialias(sine(5000, fs=48000), 48000, 8000)
        _, fs_eff, xn = decimate(x, 48000, 8000)
        self.assertLess(np.max(np.abs(xn[200:-200])), 0.01)

    def test_descriptors(self):
        f, mag = magnitude_spectrum(sine(440), FS)
        p = mag ** 2
        self.assertAlmostEqual(dominant_frequency(f, p), 440, delta=5)
        self.assertAlmostEqual(spectral_centroid(f, p), 440, delta=10)

    def test_pick_zoom_start_finds_energy(self):
        x = np.zeros(FS)
        x[20000:21000] = 1.0
        i0 = pick_zoom_start(x, FS, 0.05)
        self.assertLessEqual(abs(i0 - 20000), 2000)

    def test_effective_fs_is_fs_over_step(self):
        # La decimación solo admite factores enteros: la fs efectiva puede diferir de la pedida.
        step, fs_eff, _ = decimate(np.zeros(44100), 44100, 8000)
        self.assertEqual(step, 6)
        self.assertAlmostEqual(fs_eff, 44100 / 6)
        self.assertNotAlmostEqual(fs_eff, 8000, delta=500)

    def test_decimation_step(self):
        self.assertEqual(decimation_step(44100, 22050), 2)
        self.assertEqual(decimation_step(44100, 8000), 6)
        self.assertEqual(decimation_step(44100, 44100), 1)
        self.assertEqual(decimation_step(44100, 200000), 1)


class TestAliasingResidual(unittest.TestCase):
    def test_residual_is_smaller_with_filter(self):
        x = sine(5000, fs=48000)
        f, mag = magnitude_spectrum(x, 48000)
        fold, resid = aliasing_residual(f, mag, 48000, 8000, antialias=False)
        self.assertAlmostEqual(fold, resid)            # sin filtro se pliega todo
        fold_f, resid_f = aliasing_residual(f, mag, 48000, 8000, antialias=True)
        self.assertAlmostEqual(fold, fold_f)            # la energía que se pliega no cambia
        self.assertLess(resid_f, fold * 0.01)           # el filtro la deja pasar casi toda

    def test_nyquist_report_mentions_residual(self):
        x = sine(5000, fs=48000)
        f, mag = magnitude_spectrum(x, 48000)
        rep = nyquist_report(f, mag, 48000, 8000, 6, True, fs_asked=8000)
        self.assertFalse(rep.ok)
        self.assertIn("fs pedida 8000 Hz", rep.as_text())
        self.assertIn("aliasing residual" if False else "queda el", rep.as_text())

    def test_ok_when_above_nyquist(self):
        f, mag = magnitude_spectrum(sine(1000), FS)
        rep = nyquist_report(f, mag, FS, 22050, 2, True, fs_asked=22050)
        self.assertTrue(rep.ok)
        self.assertAlmostEqual(rep.fs_min, 2000, delta=60)


class TestQuantization(unittest.TestCase):
    def test_lsb(self):
        self.assertAlmostEqual(AdcConfig(8, 1.0).lsb, 2 / 256)
        self.assertAlmostEqual(AdcConfig(12, 3.3).lsb, 6.6 / 4096)

    def test_scale_factors(self):
        cfg = AdcConfig(8, 1.0)
        self.assertAlmostEqual(cfg.scale_adc, 1.0 / cfg.lsb)     # niveles por voltio
        self.assertAlmostEqual(cfg.scale_dac, cfg.lsb)          # voltios por nivel
        self.assertAlmostEqual(cfg.scale_adc * cfg.scale_dac, 1.0)
        self.assertAlmostEqual(cfg.vpp, 2.0)

    def test_sqnr_matches_theory_for_full_scale_sine(self):
        for n in (8, 12, 16):
            qr = quantize(sine(997.0, 0.999), AdcConfig(n, 1.0))
            self.assertAlmostEqual(qr.sqnr_db, 6.02 * n + 1.76, delta=1.0)

    def test_error_bounded_and_dac_roundtrip(self):
        cfg = AdcConfig(6, 2.5)
        qr = quantize(sine(440, 0.9), cfg)
        self.assertLessEqual(np.max(np.abs(qr.error)), cfg.lsb / 2 + 1e-12)
        np.testing.assert_allclose(dac_output(qr.codes_u, cfg), qr.volts_q)
        self.assertEqual(qr.codes_u.min() >= 0 and qr.codes_u.max() < cfg.levels, True)

    def test_binary(self):
        self.assertEqual(to_binary(5, 8), "00000101")

    def test_clipping(self):
        qr = quantize(np.array([1.5, -1.5, 0.0]), AdcConfig(8, 1.0))
        self.assertEqual(qr.clipped, 2)

    def test_level_table_matches_codes(self):
        cfg = AdcConfig(4, 1.0)
        qr = quantize(sine(440, 0.8), cfg)
        rows, note = nivel_table(cfg)
        self.assertEqual(len(rows), cfg.levels)
        self.assertIn("16 niveles", note)
        # el nivel binario coincide con el código del ADC
        for i, (lvl, volt, code) in enumerate(rows):
            self.assertEqual(int(lvl), i)
            self.assertEqual(code, to_binary(i, cfg.bits))
            self.assertAlmostEqual(float(volt), (i - cfg.levels // 2) * cfg.lsb)
        self.assertEqual(int(rows[0][0]), 0)

    def test_level_voltages_are_mid_tread(self):
        cfg = AdcConfig(8, 1.0)
        lv = level_voltages(cfg)
        self.assertEqual(len(lv), 256)
        # V(i) = (i − 2^(N−1))·Δ: hay un nivel exacto en 0 V (mid-tread)
        self.assertAlmostEqual(lv[0], -cfg.vref)
        self.assertAlmostEqual(lv[-1], cfg.vref - cfg.lsb)
        self.assertAlmostEqual(lv[cfg.levels // 2], 0.0)
        self.assertAlmostEqual(float(np.diff(lv).mean()), cfg.lsb)

    def test_pcm_stream(self):
        cfg = AdcConfig(8, 1.0)
        s = pcm_stream(np.array([0, 128, 255]), cfg, 3)
        self.assertEqual(s, "00000000 10000000 11111111")

    def test_bits_comparison_table(self):
        rows = bits_comparison(sine(440, 0.9), 1.0, [2, 4, 8])
        self.assertEqual([r["levels"] for r in rows], [4, 16, 256])
        for r in rows:
            self.assertAlmostEqual(r["sqnr_theo_db"], 6.02 * r["bits"] + 1.76)
            self.assertGreater(r["sqnr_db"], 0)


class TestSegment(unittest.TestCase):
    def test_truncation_is_reported(self):
        from core.audio_io import Audio
        aud = Audio(path="x", fs=44100, x=np.zeros(44100 * 3), channels=1, bits=16, is_float=False)
        seg = extract_segment(aud, Params(duration_s=10.0))
        self.assertTrue(seg.truncated)
        self.assertAlmostEqual(seg.actual_s, 3.0)
        self.assertAlmostEqual(seg.requested_s, 10.0)

    def test_full_segment_not_truncated(self):
        from core.audio_io import Audio
        aud = Audio(path="x", fs=44100, x=np.zeros(44100 * 30), channels=1, bits=16, is_float=False)
        seg = extract_segment(aud, Params(duration_s=10.0))
        self.assertFalse(seg.truncated)

    def test_normalization_scales_peak(self):
        from core.audio_io import Audio
        x = np.zeros(44100)
        x[100] = 0.5
        aud = Audio(path="x", fs=44100, x=x, channels=1, bits=16, is_float=False)
        seg = extract_segment(aud, Params(duration_s=1.0, normalize=True))
        self.assertAlmostEqual(float(np.max(np.abs(seg.x))), 0.99, places=6)

    def test_filter_over_shoot_does_not_saturate_the_adc(self):
        """El filtro antialiasing sobreimpulsa: el pico se normaliza DESPUÉS de filtrar."""
        from core.audio_io import Audio
        rng = np.random.default_rng(7)
        x = (rng.standard_normal(44100) * 0.2).astype(np.float32)
        x[20000] = 1.0                       # transitorio fuerte -> el filtro sobreimpulsa
        aud = Audio(path="x", fs=44100, x=x, channels=1, bits=16, is_float=False)
        p = Params(duration_s=1.0, fs_target=8000, antialias=True, bits=16)
        seg = extract_segment(aud, p)
        self.assertTrue(seg.filtered)
        # el pico de la señal YA filtrada es exactamente HEADROOM
        self.assertAlmostEqual(float(np.max(np.abs(seg.x))), 0.99, places=6)
        smp = sample_segment(seg, p)
        qr = quantize(smp.x_n, AdcConfig(16, 1.0))
        self.assertEqual(qr.clipped, 0)      # el ADC no satura

    def test_extract_without_antialias_keeps_the_original(self):
        from core.audio_io import Audio
        rng = np.random.default_rng(3)
        x = rng.standard_normal(44100).astype(np.float32)
        aud = Audio(path="x", fs=44100, x=x, channels=1, bits=16, is_float=False)
        seg = extract_segment(aud, Params(duration_s=1.0, fs_target=8000), antialias=False)
        self.assertFalse(seg.filtered)
        np.testing.assert_allclose(seg.x, seg.x_raw * seg.gain, rtol=1e-12)


class TestAudioIO(unittest.TestCase):
    def test_wav_roundtrip_header_bits(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.wav")
            wavfile.write(p, FS, (sine(440, 0.5) * 32767).astype(np.int16))
            a = load_wav(p)
            self.assertEqual((a.fs, a.bits, a.channels), (FS, 16, 1))
            self.assertAlmostEqual(float(np.max(a.x)), 0.5, delta=0.01)
            bad = os.path.join(d, "b.wav")
            with open(bad, "wb") as fh:
                fh.write(b"no soy un wav")
            with self.assertRaises(AudioError):
                load_wav(bad)

    def test_sample_segment_consistency(self):
        from core.audio_io import Audio
        aud = Audio(path="x", fs=44100, x=sine(1000).astype(np.float32), channels=1, bits=16,
                    is_float=False)
        seg = extract_segment(aud, Params(duration_s=1.0, fs_target=8000, antialias=False))
        smp = sample_segment(seg, Params(duration_s=1.0, fs_target=8000, antialias=False))
        self.assertEqual(smp.step, 6)
        self.assertAlmostEqual(smp.fs_eff, 7350.0)
        self.assertAlmostEqual(smp.t_n[1] - smp.t_n[0], 6 / 44100)


if __name__ == "__main__":
    unittest.main()
