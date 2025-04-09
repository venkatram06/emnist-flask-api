import os
import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
model = load_model('emnist_cnn_model.keras')

@app.route('/')
def home():
    return "Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img_file = request.files['image']
        img = image.load_img(img_file, target_size=(28, 28), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict using the model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        return jsonify({'predicted_class': predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
