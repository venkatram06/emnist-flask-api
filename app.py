import os
import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model (make sure this is the correct path to your model file)
model = load_model('emnist_cnn_model.keras')

@app.route('/')
def home():
    return "Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if image is present in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file part'}), 400

        img_file = request.files['image']
        
        # If no image is selected
        if img_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        try:
            img = image.load_img(img_file, target_size=(28, 28), color_mode='grayscale')
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image

            # Predict using the model
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)  # Get the predicted class label

            return jsonify({'predicted_class': predicted_class})
        
        except Exception as e:
            return jsonify({'error': f'Error processing the image: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True)
