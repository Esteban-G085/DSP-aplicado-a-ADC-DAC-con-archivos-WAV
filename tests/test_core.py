"""Pruebas del núcleo (sin GUI):  python -m unittest discover -s tests"""
import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.analysis import compute_metrics, rank_for_attention  # noqa: E402
from core.audio_io import load_wav  # noqa: E402
from core.errors import AudioError  # noqa: E402
from core.quantization import AdcConfig, dac_output, quantize, to_binary  # noqa: E402
from core.sampling import decimate, effective_bandwidth, magnitude_spectrum  # noqa: E402

FS = 44100


def sine(f, a=1.0, dur=2.0, fs=FS):
    t = np.arange(int(dur * fs)) / fs
    return a * np.sin(2 * np.pi * f * t)


class TestSampling(unittest.TestCase):
    def test_peak_frequency_and_amplitude(self):
        f, mag = magnitude_spectrum(sine(1000, 0.5), FS)
        self.assertAlmostEqual(f[np.argmax(mag)], 1000, delta=1)
        self.assertAlmostEqual(mag.max(), 0.5, delta=0.02)

    def test_bandwidth(self):
        f, mag = magnitude_spectrum(sine(3000), FS)
        self.assertLess(abs(effective_bandwidth(f, mag) - 3000), 30)

    def test_aliasing_without_filter(self):
        # 5 kHz muestreada a 8 kHz sin antialiasing -> aparece en |5000-8000| = 3000 Hz
        step, fs_eff, xn = decimate(sine(5000, fs=48000), 48000, 8000, antialias=False)
        f, mag = magnitude_spectrum(xn, fs_eff)
        self.assertEqual(step, 6)
        self.assertAlmostEqual(f[np.argmax(mag)], 3000, delta=5)

    def test_antialias_removes_it(self):
        _, fs_eff, xn = decimate(sine(5000, fs=48000), 48000, 8000, antialias=True)
        self.assertLess(np.max(np.abs(xn[200:-200])), 0.01)


class TestQuantization(unittest.TestCase):
    def test_lsb(self):
        self.assertAlmostEqual(AdcConfig(8, 1.0).lsb, 2 / 256)
        self.assertAlmostEqual(AdcConfig(12, 3.3).lsb, 6.6 / 4096)

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


class TestAudioAndAnalysis(unittest.TestCase):
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

    def test_metrics_and_ranking(self):
        low = compute_metrics("a", "grave", sine(200), FS)
        high = compute_metrics("b", "agudo", sine(6000), FS)
        self.assertAlmostEqual(low.dominant_hz, 200, delta=5)
        self.assertLess(low.centroid_hz, high.centroid_hz)
        ranking = rank_for_attention([low, high])
        self.assertEqual(ranking[0].key, "a")
        self.assertEqual(len(rank_for_attention([low])), 1)


if __name__ == "__main__":
    unittest.main()
