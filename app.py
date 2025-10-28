import joblib
import numpy as np
from flask import Flask, request, jsonify
import os

# --- Configuration (UPDATED FILE NAMES) ---
MODEL_PATH = 'retrained_logreg_model.pkl' # New model file
SCALER_PATH = 'feature_scaler_new.pkl'   # New scaler file
EXPECTED_FEATURES = 13

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Model and Scaler Globally ---
try:
    # Get the directory where the script is running
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct full paths to the files
    model_full_path = os.path.join(base_dir, MODEL_PATH)
    scaler_full_path = os.path.join(base_dir, SCALER_PATH)

    # Load the trained Logistic Regression model
    model = joblib.load(model_full_path)
    
    # Load the StandardScaler object
    scaler = joblib.load(scaler_full_path)
    
    print("✅ Logistic Regression Model and Scaler loaded successfully. Ready for predictions.")

except FileNotFoundError:
    print(f"🚨 CRITICAL ERROR: Model or scaler files not found. Ensure {MODEL_PATH} and {SCALER_PATH} are in the repository.")
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
        data = request.get_json(force=True)
        features = data.get('features')

        if features is None or len(features) != EXPECTED_FEATURES:
            return jsonify({
                "error": f"Invalid input. Expected 'features' array with {EXPECTED_FEATURES} elements."
            }), 400

        # Convert input to NumPy array, reshape for single prediction
        input_array = np.array(features).reshape(1, -1)

        # Scale the input data using the trained scaler (CRITICAL STEP)
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        result = int(prediction)

        message = "Prediction: High Risk of Heart Disease (1)" if result == 1 else "Prediction: Low Risk of Heart Disease (0)"

        # Return the result
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
