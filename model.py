# model.py

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import ModelCheckpoint

def crear_modelo(input_shape, num_units=512, dropout_rate=0.3, num_features=13):  # Ajusta num_units si es necesario
    """ Crea un modelo secuencial con capas LSTM para generación de música. """

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(num_units, return_sequences=True))
    model.add(Dropout(dropout_rate))

    # Agregar más capas LSTM si se desea
    # model.add(LSTM(num_units, return_sequences=True))
    # model.add(Dropout(dropout_rate))

    model.add(LSTM(num_units))  # Última capa LSTM sin 'return_sequences'
    model.add(Dropout(dropout_rate))

    # Cambiar la última capa a una capa densa para regresión
    model.add(Dense(num_features))  # Sin función de activación o activación lineal

    # Cambiar la función de pérdida a mean_squared_error
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def entrenar_modelo(model, X_train, y_train, epochs=25, batch_size=64):  # Aumentar epochs según sea necesario
    """ Entrena el modelo con los datos proporcionados. """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

if __name__ == "__main__":
    # Cargar datos de entrenamiento
    # X_train, y_train = cargar_tus_datos_de_entrenamiento()

    # Asumiendo que X_train e y_train ya están cargados y tienen la forma correcta
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1]  # Ajustar según la cantidad de clases/etiquetas

    modelo = crear_modelo(input_shape, num_units=256, dropout_rate=0.3, num_classes=num_classes)
    entrenar_modelo(modelo, X_train, y_train)