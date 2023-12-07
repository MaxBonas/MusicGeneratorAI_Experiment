# utils.py

import numpy as np
from keras.models import load_model

def guardar_modelo(modelo, ruta_archivo):
    """ Guarda el modelo Keras en el archivo especificado. """
    modelo.save(ruta_archivo + '.keras')
    print(f"Modelo guardado en {ruta_archivo}.keras")

def cargar_modelo(ruta_archivo):
    """ Carga un modelo Keras desde el archivo especificado. """
    modelo = load_model(ruta_archivo)
    print(f"Modelo cargado de {ruta_archivo}")
    return modelo

def guardar_secuencia(secuencia, ruta_archivo):
    """ Guarda una secuencia (array de NumPy) en un archivo .npy. """
    np.save(ruta_archivo, secuencia)
    print(f"Secuencia guardada en {ruta_archivo}")

def cargar_secuencia(ruta_archivo):
    """ Carga una secuencia desde un archivo .npy. """
    secuencia = np.load(ruta_archivo)
    print(f"Secuencia cargada de {ruta_archivo}")
    return secuencia