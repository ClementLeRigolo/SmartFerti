import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create a folder to save result files
output_dir = 'models'
os.makedirs(output_dir, exist_ok=True)

# Load the data
data = pd.read_csv('dataset.csv')

# Prepare the data
X = data[['temperature', 'humidity', 'ph', 'rainfall', 'label']]
y = data[['N', 'P', 'K']]

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the numerical features (excluding 'label')
X_scaled = X.copy()
X_scaled[['temperature', 'humidity', 'ph', 'rainfall']] = scaler.fit_transform(X[['temperature', 'humidity', 'ph', 'rainfall']])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(5,)),  # 5 input features
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(3)  # 3 output features (N, P, K)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='mean_squared_error', 
              metrics=['mae'])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train, y_train, 
                    epochs=1000, 
                    batch_size=32, 
                    validation_split=0.2,
                    callbacks=[early_stop])

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae}")

# Save the TensorFlow model in .h5 format
model.save(os.path.join(output_dir, 'smartferti.h5'))

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted TFLite model
with open(os.path.join(output_dir, 'smartferti.tflite'), 'wb') as f:
    f.write(tflite_model)

# Function to preprocess new input data
def preprocess_input(temperature, humidity, ph, rainfall, label):
    input_data = np.array([[temperature, humidity, ph, rainfall, label]])
    input_data[:, :4] = scaler.transform(input_data[:, :4])
    return input_data.astype(np.float32)

# Example usage
new_input = preprocess_input(20.87974371, 82.00274423, 6.502985292, 202.9355362, 0)
prediction = model.predict(new_input)
print("Prediction for new input:", prediction)

# Verify the TFLite model
interpreter = tf.lite.Interpreter(model_path=os.path.join(output_dir, 'smartferti.tflite'))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Use the same sample for TFLite prediction
interpreter.set_tensor(input_details[0]['index'], new_input)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])
print("TFLite model prediction:", tflite_output)