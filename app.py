import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
import io
from PIL import Image

# Define classification labels according to the model
LABELS = ["metal", "organico", "papel_y_carton", "plastico", "vidrio"]

# Load the previously trained AI model
try:
    model = keras.models.load_model("modelo_basura.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Make sure you have a 'templates/index.html' file

def preprocess_image(image):
    """Preprocess the image for AI."""
    try:
        image = image.resize((128, 128))  # Adjust size according to model
        image = np.array(image) / 255.0  # Normalization
        image = np.expand_dims(image, axis=0)  # Expand dimensions
        return image
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return None

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image has been sent"}), 400
    
    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        processed_image = preprocess_image(image)
        if processed_image is None:
            return jsonify({"error": "Error processing image"}), 500
        
        if model is None:
            return jsonify({"error": "Model is not loaded correctly"}), 500
        
        prediction = model.predict(processed_image)
        # Assuming 'prediction' has shape (1, num_classes)
        class_index = np.argmax(prediction[0])
        material = LABELS[class_index]  # Assign the corresponding label
        
        return jsonify({
            "material": material
        })
    except Exception as e:
        return jsonify({"error": f"Error in prediction: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
