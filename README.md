# Smart Fertilization Prediction Model

## Overview

This project develops and utilizes a neural network model to predict the required nutrients (Nitrogen - N, Phosphorus - P, Potassium - K) for soil based on environmental data (temperature, humidity, pH, rainfall, and the type of plant). The model is trained using TensorFlow and deployed for inference using TensorFlow Lite, which is suitable for low-power and mobile devices.
Types of plant (Rice,Maize, Chickpea; Kidney beans; pigeonpeas; mothbeans; mungbean;blackgram; lentil; pomegranate; banana; mango; grapes; watermelon; muskmelon; apple; orange;papaya; coconut; cotton; jute; coffee)
The type of plant is represented from 0 to 21 in the input.

## Requirements

### Software

- Python 3.8 or later
- TensorFlow 2.x
- scikit-learn
- pandas
- NumPy
- joblib

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/smart-fertilization-model.git
   cd smart-fertilization-model
   ```

2. Install the required Python libraries:
   ```bash
   pip install tensorflow pandas numpy scikit-learn joblib
   ```

## Project Structure

- `regression_model.py`: Script for training the neural network model and saving the scaler and model in TensorFlow and TFLite formats.
- `infer.py`: Script for loading the trained model and scaler, and performing inference.
- `models/`: Directory where trained models and scaler are saved.
- `dataset.csv`: Dataset file containing the input features and labels.

## Usage

### Training the Model

Run the `regression_model.py` script to train the model and save it along with the scaler. This will generate the necessary model files in the `models` directory.

```bash
python regression_model.py
```

### Running Inference

After training, use the `infer.py` script to perform inference using new input data. Modify the `input_data` in the script as needed to reflect new environmental conditions.

```bash
python infer.py
```

### Files Generated

- `smartferti.h5`: Trained TensorFlow model.
- `smartferti.tflite`: TensorFlow Lite model for deployment on mobile or low-power devices.
- `scaler.gz`: Scaler object used for preprocessing input data.
