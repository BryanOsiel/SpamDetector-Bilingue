# type: ignore

"""
app.py
Servidor web con Flask para exponer el modelo de spam como una API.
"""

import os
import re
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# -------------------------------------------------------
# 1. Configuración Inicial de Flask
# -------------------------------------------------------
app = Flask(__name__)
# Habilitamos CORS para permitir peticiones desde el navegador
CORS(app) 

# -------------------------------------------------------
# 2. Carga del Modelo y Tokenizer (Se hace una sola vez)
# -------------------------------------------------------
MODEL_PATH = "spam_model_bilingue.keras"
TOKENIZER_PATH = "tokenizer_config_bilingue.json"
MAX_LENGTH = 120 # ¡Debe ser el mismo que en el entrenamiento!

print("Cargando modelo...")
model = load_model(MODEL_PATH, compile=False)
print("✅ Modelo cargado.")

print("Cargando tokenizer...")
with open(TOKENIZER_PATH, 'r', encoding='utf8') as f:
    tokenizer_json_string = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json_string)
print("✅ Tokenizer cargado.")

# -------------------------------------------------------
# 3. Funciones de Predicción (Idénticas a predict.py)
# -------------------------------------------------------
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-zA-Z0-9\s'áéíóúüñ]", "", texto) 
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def predecir_mensaje(texto, threshold=0.5):
    texto_limpio = limpiar_texto(texto)
    seq = tokenizer.texts_to_sequences([texto_limpio])
    pad = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    # Realizar predicción
    pred_score = model.predict(pad, verbose=0)[0][0]
    
    # Devolver un diccionario con los resultados
    if pred_score > threshold:
        return {
            "prediction": "SPAM", 
            "score": f"{pred_score:.2f}",
            "full_text": f"🚨 SPAM (Puntuación: {pred_score:.2f})"
        }
    else:
        return {
            "prediction": "NO SPAM", 
            "score": f"{pred_score:.2f}",
            "full_text": f"✅ NO SPAM (Puntuación: {pred_score:.2f})"
        }

# -------------------------------------------------------
# 4. Definición de las Rutas (API)
# -------------------------------------------------------

@app.route("/")
def home():
    """Sirve la página web principal (index.html)."""
    # Usamos render_template, pero para eso 'index.html' debe
    # estar en una carpeta llamada 'templates'.
    # Por simplicidad, puedes abrir 'index.html' directamente desde tu disco.
    return "Servidor API en funcionamiento. Abre el archivo index.html en tu navegador."

@app.route("/predict", methods=["POST"])
def api_predict():
    """
    Endpoint de la API para predicciones.
    Espera un JSON como: { "message": "tu texto aqui" }
    Devuelve un JSON como: { "prediction": "SPAM", "score": "0.98" }
    """
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "No se proporcionó un 'message' en el JSON"}), 400
        
        mensaje_usuario = data['message']
        resultado = predecir_mensaje(mensaje_usuario)
        
        # Devolvemos el resultado como JSON
        return jsonify(resultado)

    except Exception as e:
        print(f"Error en /predict: {e}")
        return jsonify({"error": "Error interno del servidor"}), 500

# -------------------------------------------------------
# 5. Ejecutar el servidor
# -------------------------------------------------------
if __name__ == "__main__":
    print("Iniciando servidor Flask en http://127.0.0.1:5000")
    # Usamos '0.0.0.0' para que sea accesible en la red, 
    # pero '127.0.0.1' (localhost) es más seguro si solo lo usas tú.
    app.run(host='127.0.0.1', port=5000, debug=True)