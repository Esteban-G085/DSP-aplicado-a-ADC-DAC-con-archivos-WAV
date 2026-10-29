# DSP aplicado a ADC–DAC con archivos WAV

Programa de escritorio en Python (Tkinter + Matplotlib) para el trabajo de comunicaciones:
carga de audios WAV por género musical, muestreo con FFT, teorema de Nyquist-Shannon y
cuantización en un ADC–DAC ideal.

## Instalación

```bash
sudo apt install python3-tk python3-venv          # Debian / WSL2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

En Windows: `pip install -r requirements.txt` y `python main.py`.

## Estructura

```
main.py                 punto de entrada
core/                   lógica DSP, sin dependencias de interfaz
  audio_io.py           lectura de WAV (PCM 8/16/24/32 bits y float) -> (a)
  genres.py             los 5 géneros con su color
  session.py            audios cargados e investigación por género -> (b)
  pipeline.py           parámetros globales: x(t) -> segmento -> x(n)
  sampling.py           FFT, ancho de banda, Nyquist-Shannon, decimación y aliasing -> (c)
  quantization.py       ADC/DAC: Δ, factores de escala, códigos binarios, SQNR -> (d)
  errors.py             errores de usuario
gui/                    ventana y pestañas
  app.py                ventana, barra de parámetros globales y menús
  params_bar.py         Inicio, Duración, fs objetivo, Bits, ±Vref, filtros
  tab_load.py           carga de audios e investigación
  tab_sampling.py       x(t), x(n), FFT, Nyquist-Shannon
  tab_quantization.py   cuantización, DAC, SQNR, códigos binarios y trama PCM
  tab_comparison.py     espectros de los 5 géneros en una sola gráfica
  plot_tab.py           clases base (figura incrustada + barra de Matplotlib)
  plot_utils.py         utilidades de graficación
  scroll_frame.py       panel con desplazamiento vertical (texto y tablas)
tests/test_core.py      python -m unittest discover -s tests
```

## Las cuatro pestañas

| Pestaña | Apartado | Contenido |
|---|---|---|
| 0. Carga de audios | a, b | Un WAV por género + músico/orquesta/banda, obra, año e investigación breve |
| 1. Señal, muestreo y FFT | c | x(t) completa, zoom con x(n) y la escalera del DAC, espectro de magnitud \|X(f)\| y veredicto de Nyquist-Shannon |
| 2. Cuantización ADC–DAC | d | x(n) cuantizada con el error e(n) en el eje derecho, comparación de 2 a 8 bits, SQNR, códigos binarios y trama PCM |
| 3. Comparación de espectros | — | Los espectros de magnitud de los 5 géneros superpuestos en una sola gráfica, con métricas por género |

En las pestañas 1 y 2 se elige el género con el desplegable; en la 3 se comparan todos a la vez.
Para guardar una figura use el botón **Guardar** de la barra de herramientas de Matplotlib,
que ya va incrustada en cada pestaña.

### Reparto del espacio

Cada pestaña de gráficas es un panel con **dos zonas separadas por una barra arrastrable**: arriba
la figura, abajo el texto, las tablas y la trama PCM. Arrastrando la barra se decide cuánto
espacio va a cada zona, y el botón **«Solo gráficas»** de la barra de controles pliega la zona
inferior para dejar la figura a tamaño completo. Las pestañas 1 y 2 tienen la misma rejilla 2×2,
de modo que sus paneles se ven del mismo tamaño; el contenido largo (tablas, trama PCM) se recorre
con la barra de desplazamiento del panel inferior.

## Fórmulas implementadas

- **Muestreo:** x(n) = x(t = n·Ts), Ts = 1/fs. Nyquist-Shannon: fs ≥ 2·f_max, con f_max = frecuencia
  que acumula el 99 % de la energía (FFT con ventana Hann, magnitud corregida por la ganancia coherente).
- **Decimación:** solo admite factores enteros, de modo que la fs efectiva es fs/step y casi nunca
  coincide con la fs pedida (pedir 8000 Hz sobre un archivo de 44100 Hz da 7350 Hz). El programa
  siempre dice las dos cifras.
- **Filtro antialiasing:** Butterworth de orden 8 con corte en 0.45·fs_nueva, en fase cero. El
  programa no dice que el aliasing «desaparece»: mide cuánta energía se pliega y cuánta sobrevive
  al filtro, y lo expresa en dB.
- **ADC bipolar de N bits, rango ±Vref:** Δ (1 LSB) = 2·Vref/2^N; v = x·Vref [V];
  q = round(v/Δ); código ADC en offset binario = q + 2^(N−1); nivel i → V(i) = (i − 2^(N−1))·Δ.
- **Factores de escala:** ADC = 1/Δ niveles por voltio; DAC = Δ voltios por nivel.
- **DAC ideal:** V_DAC = q·Δ. Error de cuantización acotado a ±Δ/2.
- **SQNR:** de una senoide a plena escala, 6.02·N + 1.76 dB. El programa muestra además la SQNR
  medida sobre la señal y la esperada (P_señal/(Δ²/12)).

## Experimentos útiles para la presentación

- **Aliasing:** fs objetivo de x(n) = 4000 Hz y desmarque «Filtro antialiasing» (pestaña 1): el
  espectro de x(n) se eleva por el plegado. Con el filtro, el residuo medido cae de 10 % a 0.1 %.
- **Nyquist:** con fs objetivo = 22050 Hz el criterio se cumple; bajando a 8000 Hz deja de cumplirse
  y el informe lo dice con el número exacto de energía plegada.
- **Ruido de cuantización:** N = 4 vs 8 vs 16 (pestaña 2): cambian la escalera, el error y los
  6.02 dB por bit.
- **Vref:** 1.0, 3.3 o 5.0 V cambian Δ en voltios, pero no en bits.

## Notas

- El WAV estéreo se convierte a mono (promedio). Se admiten PCM de 8/16/24/32 bits y float.
- «Normalizar pico» lleva el pico a 0.99 de la escala completa del ADC. La normalización se aplica
  **después** del filtro antialiasing: un paso bajo sobreimpulsa en los transitorios (hasta un 12 %
  en audio real) y, si se normalizara antes, el pico filtrado superaría ±Vref y el ADC saturaría.
- Los apartados se calculan sobre el segmento definido por *Inicio* y *Duración* (0.5 a 60 s). Si el
  audio es más corto que la duración pedida, se analiza el audio entero y se avisa en pantalla.
