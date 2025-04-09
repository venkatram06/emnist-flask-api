from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("emnist_cnn_model.keras")

@app.route("/")
def home():
    return "EMNIST Flask API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        # Get the uploaded file
        file = request.files['file']
        
        # Read the image
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

        # Preprocess the image (resize to 28x28, normalize, reshape)
        image = cv2.resize(image, (28, 28))
        image = image / 255.0
        image = image.reshape(1, 28, 28, 1)

        # Perform prediction
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return jsonify({"predicted_class": int(predicted_class)})

    except Exception as e:
        # Handle any errors
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
