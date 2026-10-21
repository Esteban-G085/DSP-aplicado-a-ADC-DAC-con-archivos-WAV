# DSP aplicado a ADC–DAC con archivos WAV

Programa en Python con interfaz gráfica (Tkinter + Matplotlib), dividido en módulos.

## Instalación (WSL2 / Debian)

```bash
sudo apt install python3-tk python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

WSL2 con WSLg (Windows 11 o Windows 10 reciente) abre la ventana directamente.

Para probar sin música: menú **Archivo → Cargar demo sintética** (5 audios generados por código).
Para la entrega use sus propios WAV de cada género.

## Estructura

```
main.py                 punto de entrada
core/                   lógica DSP, sin interfaz (se puede probar sola)
  audio_io.py           lectura de WAV (a)
  sampling.py           FFT, ancho de banda, Nyquist-Shannon, decimación y aliasing (c)
  quantization.py       ADC/DAC: Δ, códigos, SQNR (d)
  analysis.py           frecuencia dominante, centroide, bandas, índice TDAH (e)
  pipeline.py           parámetros globales: x(t) -> segmento -> x(n)
  report.py, session.py, genres.py, synth.py, errors.py
gui/                    ventana y pestañas (una por apartado)
tests/test_core.py      python -m unittest discover -s tests
tools/make_test_wavs.py genera WAV sintéticos de prueba
```

## Qué hace cada pestaña

| Pestaña | Apartado | Contenido |
|---|---|---|
| 0. Carga | a, b | Un WAV por género + músico/orquesta/banda, obra, año e investigación breve |
| 1. x(t) | c | Señal completa (en amplitud y en V), zoom y espectrograma |
| 2. Muestreo y FFT | c | x(n) sobre x(t), espectro de magnitud con FFT, Nyquist-Shannon y aliasing |
| 3. Cuantización | d | Δ (LSB), factor de escala en V y bits, tabla con código binario de x(n), error, SQNR, DAC |
| 4. Comparación | e | Espectros de los 5 géneros en una gráfica, energía por banda, índice TDAH |
| 5. Integrado | e | Las 5 señales x(t) y sus envolventes en una sola figura |

Menú **Archivo**: exportar todas las figuras (PNG) y las métricas (CSV) para la presentación.

## Fórmulas implementadas

- Muestreo: x(n) = x(t = n·Ts), Ts = 1/fs. Nyquist-Shannon: fs ≥ 2·f_max, con f_max = frecuencia que acumula el 99 % de la energía (FFT).
- ADC bipolar de N bits, rango ±Vref: **Δ = 2·Vref / 2^N** (resolución, 1 LSB); v = x·Vref [V]; q = round(v/Δ); código = q + 2^(N−1).
- DAC ideal: V_DAC = q·Δ. Error de cuantización acotado a ±Δ/2 (sin saturación).
- SQNR de una senoide a plena escala: 6.02·N + 1.76 dB. El programa muestra además la SQNR medida y la esperada para cada señal (P_señal / (Δ²/12)); en música es menor porque no ocupa toda la escala.

## Experimentos útiles para la presentación

- **Aliasing:** fs de x(n) = 4000 Hz y desmarcar «Filtro antialiasing» (pestaña 2): el espectro de x(n) se eleva por el plegado. Con el filtro desaparece.
- **Ruido de cuantización:** N = 4 vs 8 vs 16 (pestaña 3): cambian la forma escalonada, el error y 6.02 dB por bit.
- **Vref:** 1.0, 3.3 o 5.0 V cambian Δ en voltios, pero no en bits.

## Sobre el criterio TDAH (pestaña 4)

El índice (0-100) es una **heurística didáctica** que ordena los géneros cargados entre sí con cuatro métricas
calculadas por FFT, todas donde «menos es mejor»: centroide espectral (30 %), energía sobre 4 kHz (25 %),
flujo espectral (25 %) y variación del nivel RMS (20 %). Los pesos están en `SCORE_WEIGHTS` de `core/analysis.py`.
Idea: graves/medios, pocos agudos y señal estable implican menor estimulación sensorial.
No es un criterio clínico; la respuesta de cada estudiante con TDAH a la música es individual y la evidencia es mixta.
Conviene justificar en la presentación los pesos elegidos y citar fuentes propias sobre música y atención.

## Notas

- El WAV estéreo se convierte a mono (promedio). Se admiten PCM de 8/16/24/32 bits y float.
- «Normalizar pico» lleva el pico a 0.99 de la escala completa del ADC antes de cuantizar.
- Los apartados se calculan sobre el segmento definido por *Inicio* y *Duración* (por defecto 10 s).
