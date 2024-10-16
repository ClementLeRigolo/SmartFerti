import numpy as np
import tensorflow as tf
import joblib

def load_tflite_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_input(scaler, temperature, humidity, ph, rainfall, label):
    input_data = np.array([[temperature, humidity, ph, rainfall, label]])
    input_data[:, :4] = scaler.transform(input_data[:, :4])
    return input_data.astype(np.float32)

def predict_with_tflite_model(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def main():
    tflite_model_path = 'models/smartferti.tflite'
    scaler_path = 'models/scaler.gz'

    # Load the scaler and model
    scaler = joblib.load(scaler_path)
    interpreter = load_tflite_model(tflite_model_path)

    # Example input
    input_data = preprocess_input(scaler, 20.88, 82.00, 6.50, 202.94, 0)  # Modify 'label' if needed

    # Perform prediction
    prediction = predict_with_tflite_model(interpreter, input_data)
    print("Prediction for new input:", prediction)

if __name__ == '__main__':
    main()
