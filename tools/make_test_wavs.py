"""Genera 5 WAV sintéticos de prueba:  python tools/make_test_wavs.py [carpeta]"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.synth import write_demo_wavs  # noqa: E402

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "demo_wavs"
    for k, p in write_demo_wavs(out).items():
        print(f"{k:20s} -> {p}")
