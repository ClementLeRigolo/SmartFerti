import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Créer un dossier pour sauvegarder les fichiers de résultats
output_dir = 'models'
os.makedirs(output_dir, exist_ok=True)

# Charger les données
data = pd.read_csv('dataset.csv')

# Préparer les données
X = data[['temperature', 'humidity', 'ph', 'rainfall', 'label']]
y = data[['N', 'P', 'K']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(3)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Sauvegarder le modèle TensorFlow au format .h5 dans le dossier
model.save(os.path.join(output_dir, 'smartferti.h5'))

# Convertir le modèle en TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Sauvegarder le modèle converti TFLite dans le dossier
with open(os.path.join(output_dir, 'smartferti.tflite'), 'wb') as f:
    f.write(tflite_model)

# (Optionnel) Optimiser et sauvegarder le modèle quantifié
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(os.path.join(output_dir, 'smartferti_quant.tflite'), 'wb') as f:
    f.write(tflite_model)

# Vérifier le modèle TFLite
interpreter = tf.lite.Interpreter(model_path=os.path.join(output_dir, 'smartferti.tflite'))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], X_test)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
