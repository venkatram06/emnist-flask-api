from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Load the model
model = load_model('emnist_cnn_model.keras')

@app.route('/')
def home():
    # Render the home page using an HTML template
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Process the image
    img = Image.open(file.stream)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to match the EMNIST input size
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1) / 255.0  # Normalize and reshape for the model

    # Predict the character
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  # Get the predicted class
     
    return jsonify({'prediction': str(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
