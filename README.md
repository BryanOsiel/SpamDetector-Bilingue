# Detector de Spam Bilingüe (NLP con Redes Neuronales)

Este proyecto es una aplicación web funcional que utiliza un modelo de **Red Neuronal Recurrente (LSTM Bidireccional)** para clasificar mensajes de texto como **spam** o **no spam**.

El modelo es **bilingüe** y puede procesar texto tanto en **inglés** como en **español**, gracias al uso de embeddings alineados de fastText (MUSE).

Este repositorio fue creado como el proyecto final para la materia de **Redes Neuronales Artificiales y Aprendizaje Profundo**.

## 🚀 Aplicación Web (Demo)

El proyecto incluye una interfaz web construida con Flask (backend) y HTML/CSS/JS (frontend) que consume el modelo entrenado para realizar predicciones en tiempo real.

_(¡Pega aquí una captura de pantalla de tu `index.html` funcionando!)_

## 🛠️ Características

- **Modelo Bilingüe (EN/ES):** Utiliza embeddings alineados de fastText.
- **Arquitectura de Deep Learning:** LSTM Bidireccional para captura de contexto.
- **API Web:** El modelo se sirve a través de una API RESTful con Flask (`app.py`).
- **Interfaz Interactiva:** Frontend en `index.html` para pruebas de usuario.
- **Scripts Separados:** Código modularizado para entrenamiento (`spam_detector_bilingue.py`) y predicción (`predict.py`).

---

## 🏁 Cómo Ejecutar la Demo (Recomendado)

Esta es la forma rápida de ejecutar la aplicación web usando el modelo ya entrenado.

1.  **Clonar el repositorio:**

    ```bash
    git clone [https://github.com/tu-usuario/SpamDetector-Bilingue.git](https://github.com/tu-usuario/SpamDetector-Bilingue.git)
    cd SpamDetector-Bilingue
    ```

    _(Reemplaza con tu URL)_

2.  **Crear y activar un entorno virtual:**

    ```bash
    # En Mac/Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # En Windows (CMD/PowerShell)
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Instalar las dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Iniciar el servidor Flask:**

    ```bash
    python app.py
    ```

    El servidor se iniciará en `http://127.0.0.1:5000`.

5.  **Abrir la aplicación:**
    Simplemente haz doble clic en el archivo `index.html` en tu carpeta para abrirlo en tu navegador. ¡Ya puedes probarlo!

---

## 🏋️‍♂️ Cómo Entrenar el Modelo desde Cero

Si deseas entrenar el modelo tú mismo, deberás descargar los archivos de embeddings (vectores de palabras) que son demasiado grandes para GitHub.

1.  **Descargar los Embeddings (¡Archivos grandes!):**

    - Ve a la página de [fastText Aligned Vectors](https://fasttext.cc/docs/en/aligned-vectors.html).
    - Descarga los archivos de texto (`.vec`) para **Inglés (en)** y **Español (es)**.
    - Coloca ambos archivos en la raíz de este proyecto con los nombres:
      - `wiki.en.align.vec`
      - `wiki.es.align.vec`

2.  **Instalar dependencias:**
    Asegúrate de haber seguido los pasos 2 y 3 de la sección anterior (crear entorno e instalar `requirements.txt`).

3.  **Ejecutar el script de entrenamiento:**

    ```bash
    python spam_detector_bilingue.py
    ```

    El entrenamiento comenzará. Este proceso puede tardar varios minutos y consumirá una cantidad significativa de RAM (8GB+).

4.  **Resultado:**
    Al finalizar, se crearán dos archivos nuevos, que son los que usa la demo:
    - `spam_model_bilingue.keras` (el modelo entrenado).
    - `tokenizer_config_bilingue.json` (el diccionario de palabras).

## 📊 Fuentes de Datos

El modelo fue entrenado usando una combinación de dos corpus públicos:

- **Inglés:** [SMS Spam Collection Data Set](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) de UCI.
- **Español:** [SMS Spam Corpus Spanish](https://github.com/johfl/sms-spam-corpus-spanish) de Johfl.
