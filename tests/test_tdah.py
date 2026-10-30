"""Pruebas del perfil TDAH:  python -m unittest discover -s tests"""
import os
import sys
import unittest

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.audio_io import Audio  # noqa: E402
from core.tdah import (CENTROID_BEST_HZ, ENERGY_BANDS, HIGHS_FULL_PCT, MAX_ANALYSIS_S,  # noqa: E402
                       THIRD_OCTAVE_CENTERS, WEIGHTS, analyze_track, explain, rank, score,
                       supported_bands, third_octave_edges)

FS = 44100


def tone(f, dur=3.0, fs=FS, amp=0.5):
    t = np.arange(int(dur * fs)) / fs
    return amp * np.sin(2 * np.pi * f * t)


def make_audio(x, fs=FS):
    return Audio(path="x.wav", fs=fs, x=np.asarray(x, dtype=np.float32),
                 channels=1, bits=16, is_float=False)


class TestOctaveGrid(unittest.TestCase):
    def test_centers_and_edges(self):
        self.assertEqual(len(THIRD_OCTAVE_CENTERS), 31)
        self.assertEqual(THIRD_OCTAVE_CENTERS[0], 20)
        self.assertEqual(THIRD_OCTAVE_CENTERS[-1], 20000)
        edges = third_octave_edges()
        self.assertEqual(len(edges), len(THIRD_OCTAVE_CENTERS) + 1)
        self.assertTrue(np.all(np.diff(edges) > 0))
        # Cada banda se cierra en un borde compartido con la siguiente: la
        # rejilla es contigua, sin huecos ni solapes.
        ratio = 2.0 ** (1.0 / 6.0)
        self.assertAlmostEqual(edges[0], THIRD_OCTAVE_CENTERS[0] / ratio, places=9)
        self.assertAlmostEqual(edges[-1], THIRD_OCTAVE_CENTERS[-1] * ratio, places=9)
        for i in range(1, len(edges) - 1):
            self.assertAlmostEqual(edges[i], THIRD_OCTAVE_CENTERS[i - 1] * ratio, places=9)
        # La primera banda abarca su centro ± 1/6 de octava.
        self.assertAlmostEqual(edges[1] - edges[0],
                               THIRD_OCTAVE_CENTERS[0] * (ratio - 1 / ratio), places=9)

    def test_bands_drop_top_when_above_nyquist(self):
        # A 44.1 kHz la banda de 20 kHz termina en 22.45 kHz > Nyquist.
        idx = supported_bands(FS)
        self.assertNotIn(len(THIRD_OCTAVE_CENTERS) - 1, idx)
        self.assertIn(len(THIRD_OCTAVE_CENTERS) - 2, idx)
        # Con 48 kHz cabe todas.
        self.assertEqual(len(supported_bands(48000)), len(THIRD_OCTAVE_CENTERS))

    def test_energy_bands_cover_without_gaps(self):
        self.assertEqual(ENERGY_BANDS[0][1], 20)
        self.assertEqual(ENERGY_BANDS[-1][2], 20000)
        for (_, _, hi), (_, lo, _) in zip(ENERGY_BANDS, ENERGY_BANDS[1:]):
            self.assertEqual(hi, lo)
        self.assertEqual(len(ENERGY_BANDS), 7)


class TestProfile(unittest.TestCase):
    def test_centroid_of_pure_tone(self):
        p = analyze_track(make_audio(tone(1000.0)), max_s=3.0)
        self.assertAlmostEqual(p.centroid_hz, 1000.0, delta=40)

    def test_band_percentages_sum_to_100(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(FS * 3) * 0.2
        p = analyze_track(make_audio(x), max_s=3.0)
        self.assertAlmostEqual(float(np.sum(p.band_pct)), 100.0, delta=0.01)

    def test_pct_above_4k_matches_bands(self):
        rng = np.random.default_rng(5)
        p = analyze_track(make_audio(rng.standard_normal(FS * 3) * 0.2), max_s=3.0)
        self.assertAlmostEqual(p.pct_above_4k,
                               float(p.band_pct[5] + p.band_pct[6]), delta=0.01)

    def test_octave_levels_are_relative_and_ordered(self):
        p = analyze_track(make_audio(tone(1000.0)), max_s=3.0)
        self.assertEqual(len(p.octave_db), len(p.octave_centers))
        self.assertAlmostEqual(float(np.max(p.octave_db)), 0.0, delta=1e-9)
        self.assertTrue(np.all(np.diff(p.octave_centers) > 0))

    def test_flux_and_level_are_computed(self):
        p = analyze_track(make_audio(tone(440.0, dur=4.0)), max_s=4.0)
        self.assertGreater(p.spectral_flux, 0.0)
        self.assertGreater(p.level_std_db, 0.0)
        self.assertLess(p.level_std_db, 8.0)

    def test_too_short_audio_raises(self):
        with self.assertRaises(ValueError):
            analyze_track(make_audio(np.zeros(100)), max_s=3.0)


class TestScore(unittest.TestCase):
    def _profile(self, centroid, pct_4k, flux, level):
        """Perfil sintético con las cuatro medidas fijadas."""
        from core.tdah import TdahProfile
        return TdahProfile(name="x", fs=FS, duration_s=10.0, centroid_hz=centroid,
                           pct_above_4k=pct_4k, spectral_flux=flux, level_std_db=level,
                           band_pct=np.zeros(7), octave_centers=np.zeros(1),
                           octave_db=np.zeros(1), octave_rms_db=np.zeros(1))

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(WEIGHTS), 1.0, places=9)

    def test_index_in_range(self):
        for c, h, f, l in [(100, 0, 0, 0), (3000, 10, 3, 8), (1000, 5, 1.5, 4)]:
            s = score(self._profile(c, h, f, l))
            self.assertGreaterEqual(s.index, 0.0)
            self.assertLessEqual(s.index, 100.0)

    def test_monotonicity(self):
        base = score(self._profile(1000, 5, 1.5, 4))
        self.assertGreater(score(self._profile(700, 5, 1.5, 4)).centroid, base.centroid)
        self.assertGreater(score(self._profile(1000, 2, 1.5, 4)).highs, base.highs)
        self.assertGreater(score(self._profile(1000, 5, 0.5, 4)).stable, base.stable)
        self.assertGreater(score(self._profile(1000, 5, 1.5, 2)).level, base.level)

    def test_anchors_give_full_and_zero(self):
        full = score(self._profile(CENTROID_BEST_HZ, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(full.index, 100.0, delta=1e-6)
        zero = score(self._profile(3000.0, HIGHS_FULL_PCT, 3.0, 8.0))
        self.assertAlmostEqual(zero.index, 0.0, delta=1e-6)

    def test_contributions_sum_to_index(self):
        s = score(self._profile(1000, 5, 1.5, 4))
        self.assertAlmostEqual(sum(s.contributions), s.index, places=9)
        for c, w in zip(s.contributions, WEIGHTS):
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 100.0 * w + 1e-9)


class TestRankAndExplain(unittest.TestCase):
    def _analysis(self, label, index):
        from core.tdah import GenreAnalysis, TdahProfile, TdahScore
        p = TdahProfile(name=label, fs=FS, duration_s=10.0, centroid_hz=1000,
                        pct_above_4k=5, spectral_flux=1.5, level_std_db=4,
                        band_pct=np.zeros(7), octave_centers=np.zeros(1),
                        octave_db=np.zeros(1), octave_rms_db=np.zeros(1))
        s = TdahScore(50, 50, 50, 50, index)
        return GenreAnalysis(label=label, color="#000000", profile=p, score=s)

    def test_rank_is_descending(self):
        items = [self._analysis("A", 30), self._analysis("B", 70), self._analysis("C", 50)]
        self.assertEqual([a.label for a in rank(items)], ["B", "C", "A"])

    def test_explain_mentions_caveat(self):
        text = explain([self._analysis("A", 30), self._analysis("B", 70)])
        self.assertIn("RANKING", text)
        self.assertIn("no constituye un diagnóstico médico", text.lower())
        self.assertIn("heurística", text.lower())


class TestButterworthReference(unittest.TestCase):
    """Regresión: la integración por FFT debe parecerse a un filtro paso banda."""

    def test_octave_levels_match_butterworth_bandpass(self):
        # 30 s de ruido: las bandas de 1/3 de octava tienen suficiente energía
        # y el filtro paso banda ya ha asentado su transitorio.
        rng = np.random.default_rng(3)
        x = rng.standard_normal(FS * 30) * 0.2
        audio = make_audio(x)
        profile = analyze_track(audio, max_s=30.0)

        edges = third_octave_edges()
        reference = []
        for i in supported_bands(FS):
            lo, hi = edges[i], edges[i + 1]
            sos = signal.butter(4, [lo, hi], btype="band", fs=FS, output="sos")
            y = signal.sosfiltfilt(sos, np.asarray(audio.x, dtype=np.float64))
            # Se descartan 3 s de cada extremo: el transitorio de sosfiltfilt
            # es largo en las bandas bajas y no representa el nivel real.
            trim = 3 * FS
            y = y[trim:-trim]
            reference.append(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-30))
        reference = np.array(reference)

        # La FFT integra la potencia de la banda; el filtro paso banda tiene
        # una anchura equivalente mayor, de modo que hay un offset constante.
        diff = profile.octave_rms_db - reference
        offset = float(np.mean(diff))
        self.assertLess(float(np.max(np.abs(diff - offset))), 1.5)
        self.assertGreater(float(np.corrcoef(profile.octave_rms_db, reference)[0, 1]), 0.99)


class TestCorpusRanking(unittest.TestCase):
    """El corpus de muestra debe producir el ranking del enunciado."""

    CORPUS = os.path.join(os.path.dirname(__file__), "..", "..", "music tes")
    EXPECTED = ["Salsa o Electrónica", "Cantina al hombro", "Metal",
                "Clásica - Barroca", "Reggaeton"]
    FILES = {
        "Tiburon.WAV": "Salsa o Electrónica",
        "Copita de Licor.WAV": "Cantina al hombro",
        "Decadence.WAV": "Metal",
        "The Storm.WAV": "Clásica - Barroca",
        "Gasolina.WAV": "Reggaeton",
    }

    def test_ranking_matches_expected_order(self):
        if not os.path.isdir(self.CORPUS):
            self.skipTest("corpus de muestra no disponible")
        from core.audio_io import load_wav
        from core.tdah import analyze_track
        analyses = []
        for fname, label in self.FILES.items():
            path = os.path.join(self.CORPUS, fname)
            if not os.path.exists(path):
                self.skipTest("corpus de muestra no disponible")
            audio = load_wav(path)
            profile = analyze_track(audio, max_s=MAX_ANALYSIS_S)
            analyses.append(type("A", (), {"label": label, "profile": profile,
                                          "score": score(profile)})())
        self.assertEqual([a.label for a in rank(analyses)], self.EXPECTED)


if __name__ == "__main__":
    unittest.main()
