# main.py

import data_loader
import model
import generator
import utils
import numpy as np

def preparar_datos_para_entrenamiento(caracteristicas_mfcc, longitud_secuencia=30):
    """ Prepara los datos de entrenamiento a partir de las características MFCC. """
    X, y = [], []
    for i in range(len(caracteristicas_mfcc) - longitud_secuencia):
        X.append(caracteristicas_mfcc[i:i + longitud_secuencia])
        y.append(caracteristicas_mfcc[i + longitud_secuencia])
    X = np.array(X)
    y = np.array(y)
    return X, y

def main():
    # Cargar las características MFCC desde un archivo .npy
    todas_caracteristicas_mfcc = np.load('caracteristicas_mfcc_multiples.npy')

    X_train, y_train = preparar_datos_para_entrenamiento(todas_caracteristicas_mfcc)

    # Crear el modelo
    input_shape = (X_train.shape[1], X_train.shape[2])
    modelo = model.crear_modelo(input_shape)

    # Entrenar el modelo
    model.entrenar_modelo(modelo, X_train, y_train)

    # Guardar el modelo
    utils.guardar_modelo(modelo, 'modelo_entrenado')

    # Cargar y generar música con el modelo entrenado
    modelo_cargado = utils.cargar_modelo('modelo_entrenado.keras')
    todas_caracteristicas_mfcc = np.load('caracteristicas_mfcc_multiples.npy')

    # Generar la secuencia inicial a partir de las características MFCC cargadas
    seed_sequence = generator.obtener_secuencia_inicial(todas_caracteristicas_mfcc)

    # Generar música
    musica_generada = generator.generar_musica(modelo_cargado, seed_sequence)

    # Guardar la música generada
    utils.guardar_secuencia(musica_generada, 'musica_generada.npy')

if __name__ == "__main__":
    main()