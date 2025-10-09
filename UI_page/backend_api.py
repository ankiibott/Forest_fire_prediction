import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta

# --- 1. Model Configuration (Must match your training script) ---
# CRITICAL: These constants must be defined for model loading
SEQ_LEN = 6      
HORIZONS = 3     
PATCH_H = 13     
PATCH_W = 13     
CHANNELS = 7

MODEL_PATH = "C:\\Users\\Ankit\\Downloads\\final_model.h5" # Use raw string for path
FIXED_SAMPLE_INDEX = 17 # Placeholder index corresponding to your example
START_DATE = datetime(2015, 1, 1, 0, 0, 0) # Base date for hourly offset calculation

# --- 2. Custom Layers/Functions (Required for Model Loading) ---
def slice_output_func(x):
    return x[:, :HORIZONS, :, :, :]
def squeeze_output_func(x):
    return tf.squeeze(x, axis=-1)
def slice_output_shape(input_shape):
    return (input_shape[0], HORIZONS, input_shape[2], input_shape[3], input_shape[4])
def squeeze_output_shape(input_shape):
    return input_shape[:-1] 

CUSTOM_OBJECTS = {
    'slice_output_func': slice_output_func, 'slice_output_shape': slice_output_shape,
    'squeeze_output_func': squeeze_output_func, 'squeeze_output_shape': squeeze_output_shape,
}

# --- 3. Date Mapping Function (Calculates the correct timestamps) ---

def get_sample_date_range(sample_index, start_date):
    """Calculates the date range based on a starting index (hour offset).
    This mimics the logic in your analysis script.
    """
    # Prediction starts (SEQ_LEN) hours after the start of the sample
    # Assuming each index represents one hour in the sequence
    
    # Input Window (6 hours, 0 to 5)
    input_start_hour_offset = sample_index
    input_end_hour_offset = sample_index + SEQ_LEN - 1
    
    input_start_time = start_date + timedelta(hours=input_start_hour_offset)
    input_end_time = start_date + timedelta(hours=input_end_hour_offset)
    
    # Prediction Window (3 hours, 6 to 8)
    pred_start_hour_offset = sample_index + SEQ_LEN
    pred_end_hour_offset = pred_start_hour_offset + HORIZONS - 1
    
    pred_start_time = start_date + timedelta(hours=pred_start_hour_offset)
    pred_end_time = start_date + timedelta(hours=pred_end_hour_offset)

    return {
        "inputStart": input_start_time.strftime('%H:%M:%S'),
        "inputEnd": input_end_time.strftime('%H:%M:%S'),
        "predStart": pred_start_time.strftime('%H:%M:%S'),
        "predEnd": pred_end_time.strftime('%H:%M:%S'),
        "date": input_start_time.strftime('%Y-%m-%d'), # Use start date for UI
        "full_start_date": input_start_time.strftime('%Y-%m-%d %H:%M:%S')
    }

# --- 4. Load Model (Load once when the server starts) ---
try:
    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects=CUSTOM_OBJECTS,
        safe_mode=False 
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to load model. Please check the path and dependencies.")
    print(f"Error: {e}")
    model = None 

# --- 5. Flask Application Setup ---
app = Flask(__name__)
CORS(app) 

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model failed to load on server startup."}), 500
    
    try:
        data = request.json
        # In a real scenario, you'd use these bounds for clipping
        lat_min = float(data.get('latMin'))
        lon_min = float(data.get('lonMin'))
        # ... other bounds
    except Exception as e:
        return jsonify({"error": f"Invalid input coordinates: {e}"}), 400

    # --- Time Calculation (CRITICAL FIX) ---
    time_details = get_sample_date_range(FIXED_SAMPLE_INDEX, START_DATE)

    # --- SIMULATION FOR DEMONSTRATION ---
    input_tensor = np.random.rand(1, SEQ_LEN, PATCH_H, PATCH_W, CHANNELS).astype(np.float32)

    try:
        prediction_output = model.predict(input_tensor)
        prediction_list = prediction_output[0].tolist() 
        
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500

    # --- D. Response Formatting ---
    return jsonify({
        "prediction_results": prediction_list,
        "time_details": time_details # Sending the correct time details
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
