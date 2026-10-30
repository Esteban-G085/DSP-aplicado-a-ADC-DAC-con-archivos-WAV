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
  tdah.py               1/3 de octava, reparto de energía por banda e índice heurístico -> (e)
  errors.py             errores de usuario
gui/                    ventana y pestañas
  app.py                ventana, barra de parámetros globales y menús
  params_bar.py         Inicio, Duración, fs objetivo, Bits, ±Vref, filtros
  tab_load.py           carga de audios e investigación
  tab_sampling.py       x(t), x(n), FFT, Nyquist-Shannon
  tab_quantization.py   cuantización, DAC, SQNR, códigos binarios y trama PCM
  tab_comparison.py     espectros de los 5 géneros en una sola gráfica
  tab_tdah.py           panel de 3 secciones: 1/3 de octava, bandas y perfil TDAH
  plot_tab.py           clases base (figura incrustada + barra de Matplotlib)
  plot_utils.py         utilidades de graficación
  scroll_frame.py       panel con desplazamiento vertical (texto y tablas)
tests/test_core.py      python -m unittest discover -s tests
tests/test_tdah.py      pruebas del perfil TDAH
```

## Las cinco pestañas

| Pestaña | Apartado | Contenido |
|---|---|---|
| 0. Carga de audios | a, b | Un WAV por género + músico/orquesta/banda, obra, año e investigación breve |
| 1. Señal, muestreo y FFT | c | x(t) completa, zoom con x(n) y la escalera del DAC, espectro de magnitud \|X(f)\| y veredicto de Nyquist-Shannon |
| 2. Cuantización ADC–DAC | d | x(n) cuantizada con el error e(n) en el eje derecho, comparación de 2 a 8 bits, SQNR, códigos binarios y trama PCM |
| 3. Comparación de espectros | — | Los espectros de magnitud de los 5 géneros superpuestos en una sola gráfica, con métricas por género |
| 4. Perfil TDAH | e | Nivel en bandas de 1/3 de octava (eje X logarítmico), reparto de energía en 7 bandas e índice heurístico de 0 a 100, con ranking y detalle |

En las pestañas 1 y 2 se elige el género con el desplegable; en la 3 y la 4 se comparan todos a la vez.
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
- **Bandas de 1/3 de octava:** rejilla ISO 266 de 31 centros entre 20 Hz y 20 kHz. El nivel de cada
  banda es la potencia integrada por FFT (ventana Hann) dentro del borde de la banda, y se muestra
  en dB relativos al máximo de cada género. A 44.1 kHz la banda de 20 kHz termina en 22.45 kHz,
  por encima del Nyquist, y se descarta para no dibujar una caída artificial.
- **Reparto de energía:** 7 bandas (Sub-grave, Grave, Medio-grave, Medio, Medio-agudo, Presencia,
  Brillo) que cubren 20–20000 Hz sin huecos; cada barra es el % de la energía total del género.
- **Índice heurístico TDAH:** suma ponderada de cuatro sub-puntuaciones de 0 a 100 — centroide bajo
  (30 %, escala logarítmica 500→3000 Hz), pocos agudos > 4 kHz (25 %, 0→10 %), espectro estable
  (25 %, flujo espectral 0→3.0) y nivel constante (20 %, desviación de nivel 0→8 dB).

> **Aviso:** el índice de la pestaña 4 es una **heurística didáctica**, no un instrumento clínico.
> Los pesos y las anclas son decisiones pedagógicas, no resultados de la literatura, y **no
> constituyen un diagnóstico médico**. Describe cómo se reparte la energía de la música.

## Experimentos útiles para la presentación

- **Aliasing:** fs objetivo de x(n) = 4000 Hz y desmarque «Filtro antialiasing» (pestaña 1): el
  espectro de x(n) se eleva por el plegado. Con el filtro, el residuo medido cae de 10 % a 0.1 %.
- **Nyquist:** con fs objetivo = 22050 Hz el criterio se cumple; bajando a 8000 Hz deja de cumplirse
  y el informe lo dice con el número exacto de energía plegada.
- **Ruido de cuantización:** N = 4 vs 8 vs 16 (pestaña 2): cambian la escalera, el error y los
  6.02 dB por bit.
- **Vref:** 1.0, 3.3 o 5.0 V cambian Δ en voltios, pero no en bits.
- **Perfil TDAH (pestaña 4):** con el corpus de muestra el ranking sale
  Salsa o Electrónica > Cantina al hombro > Metal > Clásica - Barroca > Reggaeton. La gráfica
  inferior derecha se ordena por índice, de modo que se lee como el ranking del texto.

## Notas

- El WAV estéreo se convierte a mono (promedio). Se admiten PCM de 8/16/24/32 bits y float.
- «Normalizar pico» lleva el pico a 0.99 de la escala completa del ADC. La normalización se aplica
  **después** del filtro antialiasing: un paso bajo sobreimpulsa en los transitorios (hasta un 12 %
  en audio real) y, si se normalizara antes, el pico filtrado superaría ±Vref y el ADC saturaría.
- Los apartados se calculan sobre el segmento definido por *Inicio* y *Duración* (0.5 a 60 s). Si el
  audio es más corto que la duración pedida, se analiza el audio entero y se avisa en pantalla.
- **La pestaña 4 es la excepción:** analiza el audio completo (hasta 60 s), ignorando *Inicio* y
  *Duración*. El flujo espectral y la variación de nivel necesitan una ventana larga para
  estabilizarse, y el perfil de un género no debe depender de un corte arbitrario: con la ventana
  de 10 s del ejemplo el ranking se invierte.
