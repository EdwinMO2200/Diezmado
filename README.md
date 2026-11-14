# Procesamiento Digital de Señales: Diezmado y Submuestreo de Audio

Repositorio enfocado en el estudio práctico del **diezmado de señales de audio** en Python, implementando reducción de tasa de muestreo, análisis energético, cálculo de FFT y filtrado antialiasing.

## Tecnologías y Librerías
* **Python**
* **NumPy** Cálculos vectorizados y manejo de arrays
* **SciPy** Diseño de filtros Butterworth y procesamiento digital
* **Librosa / SoundFile** Carga y manipulación de archivos de audio
* **Matplotlib** Visualización de espectros y respuestas en frecuencia

## Características del Proyecto
* **Carga de audio:** Procesamiento de archivos originales en mono conservando su frecuencia de muestreo nativa.
* **Filtro Antialiasing:** Implementación de un filtro pasa-bajas Butterworth de orden 5 mediante secciones segundo-orden (`SOS`) para prevenir la distorsión espectral antes de realizar el submuestreo.
* **Diezmado por factores $M=2$ y $M=4$:** Comparativa directa del comportamiento de la señal con y sin filtrado previo.
* **Análisis Espectral:** Generación de FFTs centradas y tablas comparativas de energía y potencia promedio.