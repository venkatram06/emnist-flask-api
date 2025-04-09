from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
model = load_model('emnist_cnn_model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = request.files['image'].read()
        img = image.load_img(img, target_size=(28, 28), color_mode='grayscale')
        img = image.img_to_array(img)
        img = img.reshape(1, 28, 28, 1) / 255.0  # Normalize the image

        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]

        return jsonify({'predicted_class': int(predicted_class)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
