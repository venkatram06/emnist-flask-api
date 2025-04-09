from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load the model
model = load_model('emnist_cnn_model.keras')

# Define class labels (adjust based on your dataset)
class_labels = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]

@app.route('/')
def home():
    # Serve the index.html file as the homepage
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['file']
        img = Image.open(file).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28
        img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0  # Normalize

        # Predict using the model
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]

        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
