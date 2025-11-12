# type: ignore

"""
SpamDetector BILINGÜE (ES+EN) - compatible con Python 3.8 y TensorFlow 2.13.0
Entrena, evalúa y guarda un modelo LSTM Bidireccional para detección de spam.
"""

# -------------------------------------------------------
# 1️⃣ Importaciones
# -------------------------------------------------------
import os
import re
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import requests
import zipfile
import io # Necesario para leer el CSV de español desde la URL

# -------------------------------------------------------
# 2️⃣ Carga y limpieza de datasets (Inglés + Español)
# -------------------------------------------------------

def limpiar_texto(texto):
    """Función de limpieza mejorada para incluir caracteres en español"""
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    # Permitimos letras, números, espacios, apóstrofes y caracteres en español
    texto = re.sub(r"[^a-zA-Z0-9\s'áéíóúüñ]", "", texto) 
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# --- Cargar datos en INGLÉS (el original) ---
print("📥 Descargando dataset en Inglés (sms.tsv)...")
url_en = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
data_en = pd.read_csv(url_en, sep='\t', header=None, names=['label', 'message'])

# --- Cargar datos en ESPAÑOL ---
print("📥 Descargando dataset en Español (corpus-spanish-v1.csv)...")
url_es = "https://raw.githubusercontent.com/johfl/sms-spam-corpus-spanish/master/corpus-spanish-v1.csv"
try:
    r_es = requests.get(url_es).content
    # Usamos 'latin1' ya que este CSV en particular lo requiere
    
    # 🔴 ARREGLO 1: Añadido sep=';' para leer el CSV correctamente
    data_es = pd.read_csv(io.StringIO(r_es.decode('latin1')), sep=';', usecols=['label', 'text'])
    
    data_es.rename(columns={'text': 'message'}, inplace=True) # Renombrar para que coincida
except Exception as e:
    print(f"Error al descargar o leer el dataset en español: {e}")
    data_es = pd.DataFrame(columns=['label', 'message']) # Crear DF vacío si falla

# --- Combinar y limpiar datasets ---
data = pd.concat([data_en, data_es], ignore_index=True)
data.dropna(inplace=True) # Quitar filas nulas que puedan venir del merge
data['message'] = data['message'].astype(str).apply(limpiar_texto)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Eliminar duplicados que puedan surgir de la limpieza
data = data.drop_duplicates()
# Asegurarnos de que no haya NaNs en la etiqueta después del map
data.dropna(subset=['label'], inplace=True)
data['label'] = data['label'].astype(int)

print(f"\nTotal de mensajes (EN+ES): {len(data)}")
print(data['label'].value_counts(dropna=False))


# -------------------------------------------------------
# 3️⃣ Tokenización y padding (Sin cambios)
# -------------------------------------------------------
# Aumentamos el vocabulario para dos idiomas
num_words = 20000 
tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
tokenizer.fit_on_texts(data['message'])
sequences = tokenizer.texts_to_sequences(data['message'])

max_length = 120
X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
y = np.array(data['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------
# 4️⃣ Cargar embeddings fastText (MUSE Aligned)
# -------------------------------------------------------
# fastText usa 300 dimensiones
embedding_dim = 300 
EMBEDDING_FILE_ES = "wiki.es.align.vec"
EMBEDDING_FILE_EN = "wiki.en.align.vec"

if not os.path.exists(EMBEDDING_FILE_ES) or not os.path.exists(EMBEDDING_FILE_EN):
    print("="*50)
    print("🚨 ¡ERROR: ARCHIVOS DE EMBEDDINGS NO ENCONTRADOS!")
    print(f"Por favor, descarga 'wiki.es.align.vec' y 'wiki.en.align.vec'")
    print("Desde: https://fasttext.cc/docs/en/aligned-vectors.html")
    print("Y colócalos en la misma carpeta que este script.")
    print("="*50)
    exit()

print("🧠 Cargando embeddings de fastText (puede tardar)...")
embeddings_index = {}

# 🔴 ARREGLO 2: Función 'load_fasttext_vectors' actualizada
def load_fasttext_vectors(filepath):
    """Carga los vectores de fastText manejando palabras con espacios."""
    print(f"Cargando vectores desde {filepath}...")
    with open(filepath, encoding='utf8') as f:
        # La primera línea de fastText es un header (vocab_size, dim), la saltamos
        next(f) 
        for line in f:
            values = line.split()
            
            if len(values) < (embedding_dim + 1):
                # Ignorar líneas malformadas que no tienen suficientes valores
                continue

            # La 'palabra' es todo MENOS los últimos 'embedding_dim' (300) valores
            word_parts = values[:-embedding_dim]
            word = " ".join(word_parts)
            
            # Los 'coefs' son SOLAMENTE los últimos 'embedding_dim' (300) valores
            coefs_str = values[-embedding_dim:]

            if word in tokenizer.word_index: # Optimiz: solo cargar palabras que están en nuestro corpus
                try:
                    coefs = np.asarray(coefs_str, dtype='float32')
                    embeddings_index[word] = coefs
                except ValueError:
                    # Ocasionalmente, una línea puede estar corrupta. La saltamos.
                    pass
                    
    print(f"✅ Vectores cargados desde {filepath}")

# Cargar ambos archivos en el mismo diccionario
load_fasttext_vectors(EMBEDDING_FILE_ES)
load_fasttext_vectors(EMBEDDING_FILE_EN)

print(f"Total de vectores de palabras cargados: {len(embeddings_index)}")

# --- Crear la Matriz de Embeddings ---
word_index = tokenizer.word_index
embedding_matrix = np.zeros((num_words, embedding_dim))
palabras_encontradas = 0

for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Palabras encontradas en fastText
        embedding_matrix[i] = embedding_vector
        palabras_encontradas += 1

print(f"Palabras encontradas en fastText: {palabras_encontradas} / {min(num_words, len(word_index))}")


# -------------------------------------------------------
# 5️⃣ Modelo con LSTM Bidireccional
# -------------------------------------------------------
model = Sequential([
    Input(shape=(max_length,)),
    Embedding(
        input_dim=num_words, # 20000
        output_dim=embedding_dim, # 300
        weights=[embedding_matrix],
        input_length=max_length, 
        trainable=True # Permitimos fine-tuning
    ),
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
    Bidirectional(LSTM(64, dropout=0.3)),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

# -------------------------------------------------------
# 6️⃣ Entrenamiento con EarlyStopping (Sin cambios)
# -------------------------------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------------------------------
# 7️⃣ Evaluación (Sin cambios)
# -------------------------------------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📊 Precisión en datos de prueba: {accuracy * 100:.2f}%")

# -------------------------------------------------------
# 8️⃣ Función para predecir mensajes nuevos
# -------------------------------------------------------
def predecir_mensaje(texto, threshold=0.5):
    texto = limpiar_texto(texto)
    seq = tokenizer.texts_to_sequences([texto])
    pad = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    pred = model.predict(pad, verbose=0)[0][0]
    return f"🚨 SPAM ({pred:.2f})" if pred > threshold else f"✅ NO SPAM ({pred:.2f})"

# -------------------------------------------------------
# 9️⃣ Guardar modelo y tokenizer (VERSIÓN CORREGIDA)
# -------------------------------------------------------
model.save("spam_model_bilingue.keras")
print("✅ Modelo guardado como spam_model_bilingue.keras")

# --- CAMBIO ---
# Usamos el método to_json() del tokenizer, que guarda la config completa
print("Guardando tokenizer...")
tokenizer_json = tokenizer.to_json()
with open('tokenizer_config_bilingue.json', 'w', encoding='utf8') as f:
    f.write(tokenizer_json)
print("✅ Tokenizer guardado como tokenizer_config_bilingue.json")

# -------------------------------------------------------
# 🔟 Modo interactivo (Sin cambios)
# -------------------------------------------------------
if __name__ == "__main__":
    print("\n💬 Modo interactivo bilingüe iniciado (escribe 'salir' para terminar)")
    while True:
        mensaje = input("\n✉️ Escribe un mensaje para analizar (EN o ES):\n> ")
        if mensaje.lower().strip() == "salir":
            print("👋 ¡Hasta luego!")
            break
        print("Predicción:", predecir_mensaje(mensaje))