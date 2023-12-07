# data_loader.py

import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

def cargar_multiples_archivos(directorio, limit=None):
    caracteristicas_mfcc = []
    contador = 0

    # Recorrer todas las subcarpetas en el directorio
    for subcarpeta in os.listdir(directorio):
        ruta_subcarpeta = os.path.join(directorio, subcarpeta)
        if os.path.isdir(ruta_subcarpeta):
            for archivo in os.listdir(ruta_subcarpeta):
                ruta_completa = os.path.join(ruta_subcarpeta, archivo)
                try:
                    datos, tasa_muestreo = cargar_datos_musicales(ruta_completa)
                    datos_preprocesados = preprocesar_datos(datos)
                    mfcc = extraer_caracteristicas(datos_preprocesados, tasa_muestreo)
                    caracteristicas_mfcc.append(mfcc)
                    contador += 1
                    if limit and contador >= limit:
                        return np.concatenate(caracteristicas_mfcc, axis=0)
                except Exception as e:
                    print(f"Error al procesar {ruta_completa}: {e}")
        if limit and contador >= limit:
            break

    if caracteristicas_mfcc:
        longitud_minima = min(mfcc.shape[0] for mfcc in caracteristicas_mfcc)
        caracteristicas_mfcc = [mfcc[:longitud_minima, :] for mfcc in caracteristicas_mfcc]
        return np.concatenate(caracteristicas_mfcc, axis=0)
    else:
        print(f"No se encontraron archivos válidos en {directorio} o todos los archivos causaron errores.")
        return np.array([])

def cargar_datos_musicales(ruta_archivo):
    # Carga un archivo de música utilizando librosa
    datos, tasa_muestreo = librosa.load(ruta_archivo, sr=None)
    return datos, tasa_muestreo

def extraer_caracteristicas(datos, tasa_muestreo, n_mfcc=13):
    # Extrae MFCC del archivo de audio
    mfcc = librosa.feature.mfcc(y=datos, sr=tasa_muestreo, n_mfcc=n_mfcc)
    return mfcc.T

def visualizar_forma_de_onda(datos, tasa_muestreo):
    # Visualiza la forma de onda de un archivo de audio
    plt.figure(figsize=(14, 5))
    plt.plot(np.linspace(0, len(datos) / tasa_muestreo, len(datos)), datos)  # Utiliza matplotlib para graficar
    plt.title('Forma de Onda')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.show()

def visualizar_espectrograma(datos, tasa_muestreo, n_fft=2048, hop_length=512):
    # Visualiza el espectrograma de un archivo de audio
    D = librosa.amplitude_to_db(np.abs(librosa.stft(datos, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    librosa.display.specshow(D, sr=tasa_muestreo, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma')
    plt.show()

def preprocesar_datos(datos):
    # Realiza el preprocesamiento necesario
    datos_norm = datos / np.max(np.abs(datos))
    return datos_norm

def cargar_y_procesar_un_archivo(ruta_archivo):
    datos, tasa_muestreo = cargar_datos_musicales(ruta_archivo)
    datos_preprocesados = preprocesar_datos(datos)
    mfcc = extraer_caracteristicas(datos_preprocesados, tasa_muestreo)
    return mfcc

if __name__ == "__main__":
    # Ejemplo de uso para un único archivo
    ruta_archivo_unico = 'data/comedy-detective.mp3'
    caracteristicas_mfcc_unico = cargar_y_procesar_un_archivo(ruta_archivo_unico)
    np.save('caracteristicas_mfcc_unico.npy', caracteristicas_mfcc_unico)
    print("Características MFCC de un archivo guardadas en 'caracteristicas_mfcc_unico.npy'")

    # Ejemplo de uso para múltiples archivos
    directorio_archivos = 'data/fma_small'  # Cambio a la carpeta principal
    caracteristicas_mfcc_multiples = cargar_multiples_archivos(directorio_archivos)
    np.save('caracteristicas_mfcc_multiples.npy', caracteristicas_mfcc_multiples)
    print("Características MFCC de múltiples archivos guardadas en 'caracteristicas_mfcc_multiples.npy'")