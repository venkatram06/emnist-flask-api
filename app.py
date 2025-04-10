import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# ✅ Use correct upload folder depending on environment
if os.environ.get("RENDER"):
    UPLOAD_FOLDER = '/tmp/uploads'  # Safe writable path on Render
else:
    UPLOAD_FOLDER = 'static/uploads'  # Works locally

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Mapping for EMNIST
mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

# ✅ Function to load model when needed
def load_emnist_model():
    return load_model('emnist_cnn_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # ✅ Preprocess the image
        img = Image.open(filepath).convert('L')
        img = img.resize((28, 28))
        img = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255.0

        # ✅ Load model here to prevent memory overload on Render
        model = load_emnist_model()
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        predicted_label = mapping[class_index]

        return render_template('result.html', filename=filename, label=predicted_label)

# ✅ Updated route to return image file
@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ✅ Only for Render (binds to external port)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
