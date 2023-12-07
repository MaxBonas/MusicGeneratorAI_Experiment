# generator.py

import numpy as np
import librosa
import soundfile as sf
from keras.models import load_model
import matplotlib.pyplot as plt

def cargar_modelo(filepath):
    """ Carga un modelo de Keras desde un archivo. """
    return load_model(filepath)

def generar_musica(model, seed_sequence, generation_length=500):
    """ Genera música nueva usando el modelo y una secuencia inicial. """

    generated_sequence = []
    current_sequence = seed_sequence

    for _ in range(generation_length):
        # Aquí se predice el próximo paso de la secuencia
        predicted_step = model.predict(current_sequence)
        predicted_step = predicted_step.reshape(1, 1, -1)  # Ajusta la forma si es necesario


        # Aquí se actualiza la secuencia actual con el paso predicho
        # Esto dependerá de cómo esté estructurado tu modelo y tus datos
        # Por ejemplo, puedes necesitar ajustar las dimensiones de la secuencia

        current_sequence = actualizar_secuencia(current_sequence, predicted_step)

        # Agregar el paso predicho a la secuencia generada
        generated_sequence.append(predicted_step)

    return generated_sequence

def actualizar_secuencia(current_sequence, next_step):
    # Asegúrate de que next_step tenga la forma correcta
    next_step = next_step.reshape(1, 1, -1)
    # Descarta el primer paso de la secuencia y añade el paso predicho al final
    new_sequence = np.concatenate((current_sequence[:, 1:, :], next_step), axis=1)
    return new_sequence

def obtener_secuencia_inicial(caracteristicas_mfcc, longitud_secuencia=30):
    if len(caracteristicas_mfcc) < longitud_secuencia:
        raise ValueError(f"La longitud de las características MFCC ({len(caracteristicas_mfcc)}) es menor que la longitud de secuencia requerida ({longitud_secuencia}).")

    indice_inicio = np.random.randint(0, len(caracteristicas_mfcc) - longitud_secuencia)
    secuencia_inicial = caracteristicas_mfcc[indice_inicio:indice_inicio + longitud_secuencia]
    secuencia_inicial = secuencia_inicial.reshape(1, longitud_secuencia, -1)
    return secuencia_inicial

def secuencia_a_audio(secuencia, sr=22050, max_abs_value=10):
    # Limita los valores extremos de MFCC
    if max_abs_value is not None:
        secuencia = np.clip(secuencia, -max_abs_value, max_abs_value)

    # Normalizar los MFCC para que estén en un rango más manejable
    secuencia = secuencia / max_abs_value

    # Reemplazar NaN o infinitos con ceros
    secuencia = np.nan_to_num(secuencia)

    # Convertir MFCC a audio
    audio = librosa.feature.inverse.mfcc_to_audio(np.squeeze(np.array(secuencia)), sr=sr)
    return audio


def visualizar_audio(audio):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(audio, sr=22050)
    plt.title('Forma de Onda de Audio Generado')
    plt.show()


def visualizar_mfcc(secuencia):
    # Asumiendo que la secuencia es una lista de arrays MFCC
    mfcc_array = np.squeeze(np.array(secuencia))
    plt.imshow(mfcc_array.T, aspect='auto', origin='lower')
    plt.title('Visualización de MFCC Generados')
    plt.xlabel('Paso de Tiempo')
    plt.ylabel('Coeficiente MFCC')
    plt.colorbar()
    plt.show()

def guardar_audio(audio, ruta_archivo, sr=22050):
    """ Guarda una forma de onda de audio en un archivo. """
    sf.write(ruta_archivo, audio, sr)
    print(f"Audio guardado en {ruta_archivo}")

if __name__ == "__main__":
    # Cargar el modelo
    model_filepath = 'modelo_entrenado.keras'
    model = cargar_modelo(model_filepath)
    model.summary()  # Agrega esta línea para imprimir el resumen del modelo


    # Cargar las características MFCC del archivo
    caracteristicas_mfcc = np.load('caracteristicas_mfcc_multiples.npy')

    print(f"Longitud de las características MFCC cargadas: {len(caracteristicas_mfcc)}")

    # Obtener la secuencia inicial
    seed_sequence = obtener_secuencia_inicial(caracteristicas_mfcc)
    print(f"Forma de la secuencia inicial: {seed_sequence.shape}")

    # Generar música
    generated_music = generar_musica(model, seed_sequence)

    visualizar_mfcc(generated_music)

    # Convertir la secuencia generada en audio
    generated_audio = secuencia_a_audio(generated_music)

    # Visualizar el audio generado
    visualizar_audio(generated_audio)

    # Guardar el audio generado
    guardar_audio(generated_audio, 'musica_generada.wav')

    # Puedes agregar funciones adicionales para guardar o reproducir la música generada