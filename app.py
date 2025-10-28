import joblib
import numpy as np
from flask import Flask, request, jsonify
import os

# --- Configuration (UPDATED FILE NAMES) ---
# Ensure these names match the files you pushed to GitHub
MODEL_PATH = 'retrained_logreg_model.pkl' 
SCALER_PATH = 'feature_scaler_new.pkl'   
EXPECTED_FEATURES = 13

# --- Initialize Flask App ---
# The name 'app' is critical because Gunicorn (Render's server) looks for it
app = Flask(__name__)

# --- Load Model and Scaler Globally (Simplified Path Fix) ---
# We load the files using a simple path, assuming they are in the root directory.
try:
    # Load the trained Logistic Regression model
    # Note: No os.path.join needed here, relying on direct file name access in the container root
    model = joblib.load(MODEL_PATH)
    
    # Load the StandardScaler object
    scaler = joblib.load(SCALER_PATH)
    
    print("✅ Logistic Regression Model and Scaler loaded successfully. Ready for predictions.")

except FileNotFoundError:
    print(f"🚨 CRITICAL ERROR: Model or scaler files not found. Ensure {MODEL_PATH} and {SCALER_PATH} are in the repository root.")
    # Exit, as the server cannot function without the model files.
    exit()
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making heart disease predictions.
    Receives JSON: {"features": [f1, f2, ..., f13]}
    """
    try:
        # 1. Get JSON data
        data = request.get_json(force=True)
        features = data.get('features')

        if features is None or len(features) != EXPECTED_FEATURES:
            return jsonify({
                "error": f"Invalid input. Expected 'features' array with {EXPECTED_FEATURES} elements."
            }), 400

        # 2. Prepare data
        input_array = np.array(features).reshape(1, -1)

        # 3. Scale the input data (CRITICAL STEP)
        scaled_input = scaler.transform(input_array)

        # 4. Make prediction
        prediction = model.predict(scaled_input)[0]
        result = int(prediction)

        message = "Prediction: High Risk of Heart Disease (1)" if result == 1 else "Prediction: Low Risk of Heart Disease (0)"

        # 5. Return the result
        return jsonify({
            "prediction": result,
            "message": message,
            "success": True
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            "error": "An internal server error occurred during prediction.",
            "details": "Check server logs for details."
        }), 500
