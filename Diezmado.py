import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, sosfreqz
from scipy.io import wavfile
import librosa

# ============================================================
# 1. CONFIGURACIÓN Y FUNCIONES
# ============================================================

ruta_base = r"C:\Users\edwin\OneDrive\Escritorio\Universidad Central\2025-2\Procesamiento digital de senales\Taller diezmado - copia"
nombre_audio = "SinTi.mp3"
ruta_audio = os.path.join(ruta_base, nombre_audio)

def cargar_audio(ruta):
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"\nNo se encontró el archivo:\n{ruta}\n")
    x, fs = librosa.load(ruta, sr=None, mono=True)
    return int(fs), x.astype(np.float64)

def calcular_energia_potencia(x):
    N = len(x)
    energia = np.sum(np.abs(x) ** 2)
    potencia = energia / N if N > 0 else 0
    return energia, potencia

def aplicar_filtro_antialiasing(x, fs, M):
    fs_nueva = fs / M
    fc = fs_nueva / 2
    wn = fc / (fs / 2)
    sos = butter(5, wn, btype="lowpass", output="sos")
    return sosfiltfilt(sos, x), fc, sos

def diezmar_senal(x, M):
    return x[::M]

def calcular_fft(x, fs):
    N = len(x)
    X = np.fft.fft(x)
    X_shift = np.fft.fftshift(X)
    frecuencia_hz = np.fft.fftshift(np.fft.fftfreq(N, d=1 / fs))
    frecuencia_rad = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N))
    return X_shift, frecuencia_hz, frecuencia_rad

def guardar_audio(nombre, x, fs):
    ruta = os.path.join(ruta_base, nombre)
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    maximo = np.max(np.abs(x))
    x_norm = x / maximo if maximo > 0 else x
    x_int16 = (np.clip(x_norm, -1, 1) * 32767).astype(np.int16)
    wavfile.write(ruta, int(fs), x_int16)
    print(f"Archivo guardado: {ruta}")

# ============================================================
# 2. PROCESAMIENTO DEL AUDIO ORIGINAL
# ============================================================

fs_original, x_original = cargar_audio(ruta_audio)
energia_original, potencia_original = calcular_energia_potencia(x_original)

print("\n========================================")
print("INFORMACIÓN DEL AUDIO ORIGINAL")
print("========================================")
print(f"Archivo: {nombre_audio}\nFrecuencia: {fs_original} Hz\nMuestras: {len(x_original)}\nDuración: {len(x_original) / fs_original:.2f} s")
print(f"Energía = {energia_original:.6f}\nPotencia promedio = {potencia_original:.6f}")

# ============================================================
# 3. PROCESAMIENTO AUTOMATIZADO PARA M = 2 Y M = 4
# ============================================================

resultados = {}

for M in [2, 4]:
    fs_M = fs_original / M
    x_sin = diezmar_senal(x_original, M)
    x_filtrada, fc, sos = aplicar_filtro_antialiasing(x_original, fs_original, M)
    x_con = diezmar_senal(x_filtrada, M)
    
    resultados[M] = {
        "fs": fs_M, "fc": fc, "sos": sos,
        "sin": x_sin, "con": x_con,
        "energia_sin": calcular_energia_potencia(x_sin)[0],
        "potencia_sin": calcular_energia_potencia(x_sin)[1],
        "energia_con": calcular_energia_potencia(x_con)[0],
        "potencia_con": calcular_energia_potencia(x_con)[1],
    }
    
    guardar_audio(f"audio_M{M}_sin_filtro.wav", x_sin, fs_M)
    guardar_audio(f"audio_M{M}_con_filtro.wav", x_con, fs_M)

print("\n========================================")
print("DIEZMADO - FRECUENCIAS")
print("========================================")
for M in [2, 4]:
    print(f"\nM = {M}\nFrecuencia nueva = {resultados[M]['fs']} Hz\nFrecuencia de corte = {resultados[M]['fc']} Hz")

# ============================================================
# 4. VERIFICACIÓN DE ARCHIVOS WAV
# ============================================================

print("\n========================================")
print("VERIFICACIÓN DE ARCHIVOS WAV")
print("========================================")
archivos = [f"audio_M{M}_{tipo}.wav" for M in [2, 4] for tipo in ["sin_filtro", "con_filtro"]]

for archivo in archivos:
    fs_arch, datos_arch = wavfile.read(os.path.join(ruta_base, archivo))
    print(f"\n{archivo}\nFrecuencia: {fs_arch} Hz\nMuestras: {len(datos_arch)}\nDuración: {len(datos_arch) / fs_arch:.2f} s\nMáx: {np.max(datos_arch)} | Mín: {np.min(datos_arch)}")

# ============================================================
# 5. TABLA DE RESULTADOS
# ============================================================

print("\n========================================")
print("TABLA DE ENERGÍA Y POTENCIA")
print("========================================")
print(f"{'Señal':25}{'M':>5}{'Filtro':>12}{'Muestras':>12}{'Energía':>15}{'Potencia':>15}")
print("-" * 84)
print(f"{'Original':25}{1:>5}{'No':>12}{len(x_original):>12}{energia_original:>15.6f}{potencia_original:>15.6f}")

for M in [2, 4]:
    for con_filtro, etiqueta in [(False, 'No'), (True, 'Sí')]:
        clave = 'con' if con_filtro else 'sin'
        x_ref = resultados[M][clave]
        print(f"{f'M{M}':25}{M:>5}{etiqueta:>12}{len(x_ref):>12}{resultados[M][f'energia_{clave}']:>15.6f}{resultados[M][f'potencia_{clave}']:>15.6f}")

# ============================================================
# 6. CÁLCULO DE FFTs
# ============================================================

X_orig, f_orig_hz, f_orig_rad = calcular_fft(x_original, fs_original)

fft_resultados = {}
for M in [2, 4]:
    for tipo in ['sin', 'con']:
        X_s, f_hz, f_rad = calcular_fft(resultados[M][tipo], resultados[M]['fs'])
        fft_resultados[f"M{M}_{tipo}"] = (X_s, f_hz, f_rad)

# ============================================================
# 7. GRÁFICAS
# ============================================================

# Espectro original Hz
plt.figure(figsize=(12, 5))
plt.plot(f_orig_hz, np.abs(X_orig))
plt.title("Espectro de la señal original")
plt.xlabel("Frecuencia física [Hz]")
plt.ylabel("|X(f)|")
plt.grid(True)
plt.xlim(-fs_original / 2, fs_original / 2)
plt.tight_layout()
plt.show()

# Espectro original rad/muestra
plt.figure(figsize=(12, 5))
plt.plot(f_orig_rad, np.abs(X_orig))
plt.title("Espectro de la señal original")
plt.xlabel("Frecuencia digital [rad/muestra]")
plt.ylabel("|X(e^jw)|")
plt.grid(True)
plt.xlim(-np.pi, np.pi)
plt.tight_layout()
plt.show()

# Comparación de los cuatro espectros
plt.figure(figsize=(12, 7))
for M in [2, 4]:
    for tipo, etiqueta in [('sin', f'M={M} sin filtro'), ('con', f'M={M} con filtro')]:
        _, f_hz, _ = fft_resultados[f"M{M}_{tipo}"]
        plt.plot(f_hz, np.abs(fft_resultados[f"M{M}_{tipo}"][0]), label=etiqueta)
plt.title("Comparación de espectros después del diezmado")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Espectros individuales y respuestas de filtro
for M in [2, 4]:
    for tipo, desc in [('sin', 'Sin filtro'), ('con', 'Con filtro antialiasing')]:
        plt.figure(figsize=(12, 5))
        _, f_hz, _ = fft_resultados[f"M{M}_{tipo}"]
        plt.plot(f_hz, np.abs(fft_resultados[f"M{M}_{tipo}"][0]))
        plt.title(f"M = {M} - {desc}")
        plt.xlabel("Frecuencia [Hz]")
        plt.ylabel("|X(f)|")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Respuesta del filtro
    f_filtro, resp = sosfreqz(resultados[M]['sos'], worN=4096, fs=fs_original)
    plt.figure(figsize=(12, 5))
    plt.plot(f_filtro, 20 * np.log10(np.maximum(np.abs(resp), 1e-12)))
    plt.axvline(resultados[M]['fc'], linestyle="--", label=f"fc = {resultados[M]['fc']:.2f} Hz")
    plt.title(f"Respuesta en frecuencia del filtro antialiasing - M = {M}")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud [dB]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# 8. INFORMACIÓN FINAL
# ============================================================

print("\n========================================")
print("PROCESO TERMINADO")
print("========================================")
print("\nArchivos generados:")
for archivo in archivos:
    print(os.path.join(ruta_base, archivo))
print(f"\nFrecuencias:\nOriginal : {fs_original} Hz\nM = 2    : {resultados[2]['fs']} Hz (fc = {resultados[2]['fc']} Hz)\nM = 4    : {resultados[4]['fs']} Hz (fc = {resultados[4]['fc']} Hz)")