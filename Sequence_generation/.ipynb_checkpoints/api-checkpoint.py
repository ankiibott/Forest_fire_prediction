import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# --- 1. Model Configuration ---
# Match these constants to your trained model's architecture
SEQ_LEN = 6      
HORIZONS = 3     
PATCH_H = 13     
PATCH_W = 13     
CHANNELS = 7     

# Define custom objects required for loading the model, especially Lambda layers
def slice_output_func(x): 
    # This must match the original function logic if used in the Lambda layer
    return x[:, :HORIZONS, :, :, :]

def squeeze_output_func(x): 
    # This must match the original function logic if used in the Lambda layer
    return tf.squeeze(x, axis=-1)

CUSTOM_OBJECTS = {
    'slice_output_func': slice_output_func, 
    'squeeze_output_func': squeeze_output_func,
}

MODEL_PATH = r"C:\Users\Ankit\Downloads\final_model.h5"
MODEL = None

# --- 2. Flask Setup ---
app = Flask(__name__)
# Enable CORS to allow your frontend (React/Browser) to make requests to this server
CORS(app) 

def load_model():
    """Load the trained model into memory when the server starts."""
    global MODEL
    print(f"Loading model from: {MODEL_PATH}")
    try:
        # Load the model with custom objects
        MODEL = tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects=CUSTOM_OBJECTS, 
            safe_mode=False
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR loading model: {e}")
        # If the model fails to load, the server cannot function.
        MODEL = None
        
load_model() # Load the model immediately on server startup


# --- 3. Prediction Endpoint ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts the 4D input tensor for a single sequence and returns the 
    3D predicted probability map.
    """
    if MODEL is None:
        return jsonify({"error": "Model failed to load on startup."}), 500

    try:
        # Get JSON data from the request body
        data = request.get_json()
        input_data = data.get('input_tensor')

        if not input_data:
            return jsonify({"error": "Missing 'input_tensor' field in request body."}), 400

        # 3a. Convert List (JSON) to NumPy Array
        input_array = np.array(input_data, dtype=np.float32)

        # 3b. Validate the 4D shape (Time, H, W, C)
        expected_shape = (SEQ_LEN, PATCH_H, PATCH_W, CHANNELS)
        if input_array.shape != expected_shape:
            return jsonify({
                "error": "Input shape mismatch.",
                "received_shape": input_array.shape,
                "expected_shape": expected_shape
            }), 400
        
        # 3c. Add the Batch Dimension (required by Keras)
        input_tensor_5D = np.expand_dims(input_array, axis=0) # Shape: (1, 6, 13, 13, 7)

        # 3d. Run Prediction
        # The output shape will be (1, HORIZONS, PATCH_H, PATCH_W)
        predictions_raw = MODEL.predict(input_tensor_5D, verbose=0)
        
        # 3e. Convert output to standard list format (remove batch dimension)
        output_data = predictions_raw[0].tolist()

        return jsonify({
            "status": "success",
            "predicted_probabilities": output_data, # Shape: (3, 13, 13)
            "output_shape": predictions_raw[0].shape
        })

    except Exception as e:
        # Catch any runtime errors (e.g., memory, unexpected data)
        print(f"Prediction error: {e}")
        return jsonify({"error": f"An internal error occurred during prediction: {str(e)}"}), 500

# --- 4. Server Run ---
if __name__ == '__main__':
    # You can change the port if needed, but 5000 is standard for Flask
    app.run(host='0.0.0.0', port=5000) 
