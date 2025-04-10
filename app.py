import os
from flask import Flask, render_template, request
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

# ✅ EMNIST Balanced Mapping (47 classes)
mapping = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")

# ✅ Load model
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

        # ✅ Preprocess image to match EMNIST format
        img = Image.open(filepath).convert('L').resize((28, 28))
        img = np.array(img)

        # EMNIST expects white text on black background, flipped and rotated
        img = 255 - img
        img = np.rot90(img, k=1)
        img = np.flip(img, axis=0)

        img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0

        # ✅ Predict
        model = load_emnist_model()
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        predicted_label = mapping[class_index]

        return render_template('result.html', filename=filename, label=predicted_label)

@app.route('/display/<filename>')
def display_image(filename):
    return f'<img src="/static/uploads/{filename}" width="200">'

# ✅ Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
