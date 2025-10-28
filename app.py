import joblib
import numpy as np
from flask import Flask, request, jsonify
import os

# --- Configuration ---
# NOTE: These filenames MUST match the files in your GitHub repository
MODEL_PATH = 'heart_disease_model.pkl'
SCALER_PATH = 'feature_scaler.pkl'
EXPECTED_FEATURES = 13

# --- Initialize Flask App ---
# The name 'app' is critical because Gunicorn (Render's server) looks for it
app = Flask(__name__)

# --- Load Model and Scaler Globally ---
# Load the model and scaler only ONCE when the server starts.
# This ensures fast prediction times for all subsequent requests.
try:
    # Get the directory where the script is running
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct full paths to the files
    model_full_path = os.path.join(base_dir, MODEL_PATH)
    scaler_full_path = os.path.join(base_dir, SCALER_PATH)

    # Load the trained Gradient Boosting Classifier model
    model = joblib.load(model_full_path)
    
    # Load the StandardScaler object
    scaler = joblib.load(scaler_full_path)
    
    print("✅ Model and Scaler loaded successfully. Ready for predictions.")

except FileNotFoundError:
    # If files are missing, the server cannot start.
    print(f"🚨 CRITICAL ERROR: Model or scaler files not found in the path.")
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
        # 1. Get JSON data from the request
        data = request.get_json(force=True)
        features = data.get('features')

        if features is None or len(features) != EXPECTED_FEATURES:
            return jsonify({
                "error": f"Invalid input. Expected 'features' array with {EXPECTED_FEATURES} elements."
            }), 400

        # 2. Convert input to NumPy array, reshape for single prediction
        input_array = np.array(features).reshape(1, -1)

        # 3. Scale the input data using the trained scaler (CRITICAL STEP)
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

# NOTE: We do NOT use 'if __name__ == "__main__": app.run()' here. 
# Render uses the gunicorn command 'gunicorn app:app' to start the server.
