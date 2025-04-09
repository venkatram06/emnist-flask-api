from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "emnist_cnn_model.keras"
if not os.path.exists(MODEL_PATH):
    raise Exception(f"Model file not found at {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/")
def home():
    return "EMNIST Flask API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']

        # Check if the file is an image (simple validation)
        if file.filename == '' or not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "Invalid file format. Please upload an image file (png, jpg, jpeg)."}), 400
        
        # Read the image
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        if image is None:
            return jsonify({"error": "Unable to decode image"}), 400

        # Preprocess the image (resize to 28x28, normalize, reshape)
        image = cv2.resize(image, (28, 28))
        image = image / 255.0
        image = image.reshape(1, 28, 28, 1)

        # Perform prediction
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return jsonify({"predicted_class": int(predicted_class)})

    except Exception as e:
        # Handle any errors and provide more detailed error information
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=False)  # Set debug=False in production
