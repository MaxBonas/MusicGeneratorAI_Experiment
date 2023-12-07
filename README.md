ENG - ESP

# Music Generator AI

## Description
The Music Generator AI project is an advanced application that uses machine learning to generate new music. This project leverages the processing of MFCC (Mel-Frequency Cepstral Coefficients) audio features and utilizes recurrent neural networks, specifically LSTM (Long Short-Term Memory), to generate music sequences.

## Features
- Load and process music files to extract MFCC features.
- Use Deep Learning models to generate music sequences.
- Visualize audio waveforms and spectrograms.
- Ability to generate and save generated audio.

## Requirements
- Python 3.x
- Librosa
- NumPy
- Matplotlib
- Keras
- SoundFile

## Installation
Describe how to set up and run your project. Include commands that need to be run. For example:

```bash
pip install librosa numpy matplotlib keras soundfile
```

## Usage
Explain how to use your project. For example:

1. **Loading Data**: Use `data_loader.py` to load and process music files.

    ```python
    from data_loader import cargar_y_procesar_un_archivo
    data = cargar_y_procesar_un_archivo('path/to/your/file.mp3')
    ```

2. **Generating Music**: Use `generator.py` to generate new music based on an initial sequence.

    ```python
    from generator import generar_musica
    new_music = generar_musica(model, seed_sequence)
    ```

3. **Visualization**: Use functions in `data_loader.py` and `generator.py` to visualize data and results.

    ```python
    from data_loader import visualizar_forma_de_onda
    visualizar_forma_de_onda(data)
    ```

## Project Structure
Briefly describe the structure of your project, for example:

- `data_loader.py`: Loads and processes music files.
- `generator.py`: Contains logic for generating new music.
- `model.py`: Defines the machine learning model.
- `utils.py`: Helper functions for saving and loading models and sequences.
- `main.py`: The main script for training models and generating music.

## Contributing
If you would like to contribute to the project, please consider the following:

1. Fork the repository.
2. Create a new branch (`git checkout -b new-feature`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin new-feature`).
5. Open a Pull Request.


# Music Generator AI

## Descripción
El proyecto Music Generator AI es una aplicación avanzada que utiliza aprendizaje automático para generar música nueva. Este proyecto se basa en el procesamiento de características de audio MFCC (Mel-Frequency Cepstral Coefficients) y utiliza redes neuronales recurrentes, específicamente LSTM (Long Short-Term Memory), para generar secuencias de música.

## Características
- Carga y procesamiento de archivos de música para extraer características MFCC.
- Utilización de modelos de Deep Learning para generar secuencias de música.
- Visualización de formas de onda y espectrogramas de audio.
- Capacidad para generar y guardar audio generado.

## Requisitos
- Python 3.x
- Librosa
- NumPy
- Matplotlib
- Keras
- SoundFile

## Instalación
Describa cómo configurar y ejecutar su proyecto. Incluya comandos que se deben ejecutar. Por ejemplo:

```bash
pip install librosa numpy matplotlib keras soundfile
```

## Uso
Explique cómo usar su proyecto. Por ejemplo:

1. **Cargar Datos**: Use `data_loader.py` para cargar y procesar archivos de música.

    ```python
    from data_loader import cargar_y_procesar_un_archivo
    data = cargar_y_procesar_un_archivo('ruta/a/tu/archivo.mp3')
    ```

2. **Generar Música**: Utilice `generator.py` para generar música nueva basada en una secuencia inicial.

    ```python
    from generator import generar_musica
    new_music = generar_musica(model, seed_sequence)
    ```

3. **Visualización**: Use funciones en `data_loader.py` y `generator.py` para visualizar datos y resultados.

    ```python
    from data_loader import visualizar_forma_de_onda
    visualizar_forma_de_onda(data)
    ```

## Estructura del Proyecto
Describa brevemente la estructura de su proyecto, por ejemplo:

- `data_loader.py`: Carga y procesa archivos de música.
- `generator.py`: Contiene lógica para generar música nueva.
- `model.py`: Define el modelo de aprendizaje automático.
- `utils.py`: Funciones auxiliares para guardar y cargar modelos y secuencias.
- `main.py`: Script principal para entrenar modelos y generar música.

## Contribuir

Si desea contribuir al proyecto, considere lo siguiente:

1. Realice un fork del repositorio.
2. Cree una nueva rama (`git checkout -b feature-nueva`).
3. Realice sus cambios y haga commit (`git commit -am 'Añadir nueva característica'`).
4. Haga push a la rama (`git push origin feature-nueva`).
5. Abra una Pull Request.

