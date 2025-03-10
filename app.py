import cv2
import numpy as np
import tensorflow.lite as tflite
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="food_spoilage_model.tflite")
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_names = ["Fresh", "Stale"]

# Preprocessing function
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = image.astype(np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction function
def predict(image):
    image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)
    return class_names[predicted_class]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_image():
    file = request.files["file"]
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    prediction = predict(image)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
