# type: ignore

"""
predict.py (VERSIÓN CORREGIDA)
Carga el modelo bilingüe entrenado (spam_model_bilingue.keras) 
y el tokenizer (tokenizer_config_bilingue.json) para 
realizar predicciones en nuevos mensajes.
"""

import os
import re
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------
# 1️⃣ Constantes y carga de archivos (SECCIÓN CORREGIDA)
# -------------------------------------------------------
MODEL_PATH = "spam_model_bilingue.keras"
TOKENIZER_PATH = "tokenizer_config_bilingue.json"
MAX_LENGTH = 120 # Debe ser el mismo valor usado en el entrenamiento

# --- Cargar Modelo ---
if not os.path.exists(MODEL_PATH):
    print(f"🚨 Error: No se encontró el archivo del modelo en {MODEL_PATH}")
    exit()

print("Cargando modelo...")
model = load_model(MODEL_PATH, compile=False) 
print("✅ Modelo cargado.")

# --- Cargar Tokenizer ---
if not os.path.exists(TOKENIZER_PATH):
    print(f"🚨 Error: No se encontró el archivo del tokenizer en {TOKENIZER_PATH}")
    exit()

print("Cargando tokenizer...")
try:
    with open(TOKENIZER_PATH, 'r', encoding='utf8') as f:
        # --- INICIO DEL CAMBIO ---
        # Leemos el archivo como un string de texto
        tokenizer_json_string = f.read()
        # Recreamos el tokenizer directamente desde el string JSON
        tokenizer = tokenizer_from_json(tokenizer_json_string)
        # --- FIN DEL CAMBIO ---
    print("✅ Tokenizer cargado.")

except json.JSONDecodeError:
    print(f"🚨 Error: El archivo {TOKENIZER_PATH} está corrupto o vacío.")
    print("Por favor, vuelve a ejecutar el script 'spam_detector_bilingue.py' para generarlo.")
    exit()
except Exception as e:
    print(f"Ocurrió un error inesperado al cargar el tokenizer: {e}")
    exit()


# -------------------------------------------------------
# 2️⃣ Funciones auxiliares (Sin cambios)
# -------------------------------------------------------

def limpiar_texto(texto):
    """Limpia el texto de entrada (debe ser idéntica a la del script de entrenamiento)."""
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    # Permitimos letras, números, espacios, apóstrofes y caracteres en español
    texto = re.sub(r"[^a-zA-Z0-9\s'áéíóúüñ]", "", texto) 
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def predecir_mensaje(texto, threshold=0.5):
    """Procesa un mensaje y devuelve la predicción del modelo."""
    texto_limpio = limpiar_texto(texto)
    
    # Convertir a secuencia y aplicar padding
    seq = tokenizer.texts_to_sequences([texto_limpio])
    pad = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    # Realizar predicción
    pred = model.predict(pad, verbose=0)[0][0]
    
    # Devolver resultado
    if pred > threshold:
        return f"🚨 SPAM (Puntuación: {pred:.2f})"
    else:
        return f"✅ NO SPAM (Puntuación: {pred:.2f})"

# -------------------------------------------------------
# 3️⃣ Modo interactivo (Sin cambios)
# -------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("💬 Modo de Predicción Bilingüe (EN/ES)")
    print("     (Escribe 'salir' para terminar)")
    print("="*50)
    
    while True:
        mensaje = input("\n✉️ Escribe un mensaje para analizar:\n> ")
        if mensaje.lower().strip() == "salir":
            print("👋 ¡Hasta luego!")
            break
        
        # Obtener y mostrar la predicción
        resultado = predecir_mensaje(mensaje)
        print("Predicción:", resultado)