import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load the model (make sure this file is added to your GitHub repo for Render)
model = load_model('emnist_cnn_model.keras')

# ✅ Mapping for EMNIST
mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

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

        # ✅ Preprocess
        img = Image.open(filepath).convert('L')
        img = img.resize((28, 28))
        img = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255.0

        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        predicted_label = mapping[class_index]

        return render_template('result.html', filename=filename, label=predicted_label)

@app.route('/display/<filename>')
def display_image(filename):
    return f'<img src="/static/uploads/{filename}" width="200">'

# ✅ Only for Render (binds to external port)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render uses PORT env var
    app.run(host="0.0.0.0", port=port)
