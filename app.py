import os
import numpy as np
import tensorflow.lite as tflite
import cv2
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load TFLite model
MODEL_PATH = "food_spoilage_model.tflite"

try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading TFLite model: {e}")

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define labels
labels = ["Fresh", "Stale"]

def preprocess_image(image):
    """Preprocess image for the model"""
    image = cv2.resize(image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))  # Resize to model input size
    image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0  # Normalize
    return image

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Web UI

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = preprocess_image(image)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Get prediction
        prediction = labels[np.argmax(output)]
        confidence = float(np.max(output))

        return jsonify({"prediction": prediction, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns PORT dynamically
    app.run(host="0.0.0.0", port=port)
