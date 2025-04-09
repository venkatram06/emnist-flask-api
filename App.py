from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)
model = load_model("emnist_cnn_model.keras")  # Load your model

@app.route("/")
def home():
    return "EMNIST Flask API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive base64-encoded image
        data = request.json['image']
        image = Image.open(io.BytesIO(base64.b64decode(data))).convert('L').resize((28, 28))
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))

        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)})
