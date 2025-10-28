import joblib
import numpy as np
from flask import Flask, request, jsonify
import os

# --- Configuration (UPDATED FILE NAMES AND FEATURE COUNT) ---
# Ensure these names match the files you pushed (L2 versions)
MODEL_PATH = 'retrained_logreg_model_l2.pkl'
SCALER_PATH = 'feature_scaler_new_l2.pkl'
EXPECTED_FEATURES = 13 # Assuming your heart disease model still requires 13 features

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Model and Scaler Globally ---
try:
    # Load the retrained Logistic Regression model using joblib
    model = joblib.load(MODEL_PATH)
    
    # Load the StandardScaler object using joblib
    scaler = joblib.load(SCALER_PATH)
    
    print("✅ Logistic Regression Model (L2) and Scaler loaded successfully. Ready for predictions.")

except FileNotFoundError:
    print(f"🚨 CRITICAL ERROR: Model or scaler files not found. Ensure {MODEL_PATH} and {SCALER_PATH} are in the repository root.")
    # Set to None so the server can run, but predictions will fail gracefully
    model = None
    scaler = None
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    model = None
    scaler = None


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions.
    Receives JSON: {"features": [f1, f2, ..., f13]}
    """
    # 0. Check if model is loaded
    if model is None or scaler is None:
        return jsonify({
            "error": "Model not loaded. Check server logs for file location issues.",
            "success": False
        }), 503 # Service Unavailable

    try:
        # 1. Get JSON data
        data = request.get_json(force=True)
        features = data.get('features')

        # 2. Input validation
        if features is None or len(features) != EXPECTED_FEATURES:
            return jsonify({
                "error": f"Invalid input. Expected 'features' array with {EXPECTED_FEATURES} elements.",
                "success": False
            }), 400

        # 3. Prepare data and scale (CRITICAL STEP)
        input_array = np.array(features, dtype=float).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        # 4. Make prediction
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]
        result = int(prediction)

        # 5. Format response message
        message = "Prediction: High Risk (1)" if result == 1 else "Prediction: Low Risk (0)"
        confidence = prediction_proba[result] if result < len(prediction_proba) else None
        
        # 6. Return the result
        response = {
            "prediction": result,
            "message": message,
            "confidence": f"{confidence*100:.2f}%" if confidence is not None else "N/A",
            "success": True
        }
        
        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            "error": "An internal server error occurred during prediction.",
            "details": str(e),
            "success": False
        }), 500

if __name__ == '__main__':
    # This block is usually for local testing. Use 'gunicorn' or 'flask run' for production.
    # app.run(debug=True)
    pass
