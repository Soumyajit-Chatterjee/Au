import joblib
import numpy as np
from flask import Flask, request, jsonify
import os

# --- Configuration (UPDATED FILE NAMES FOR SMOTE MODEL) ---
# New files resulting from SMOTE training:
MODEL_PATH = 'retrained_logreg_model_smote.pkl'
SCALER_PATH = 'feature_scaler_smote.pkl'
EXPECTED_FEATURES = 13 # Assuming your model still requires 13 features

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Model and Scaler Globally ---
try:
    # Load the retrained Logistic Regression model (SMOTE version)
    model = joblib.load(MODEL_PATH)
    
    # Load the StandardScaler object (SMOTE version)
    scaler = joblib.load(SCALER_PATH)
    
    print(f"✅ SMOTE-trained Logistic Regression Model and Scaler loaded successfully from {MODEL_PATH} and {SCALER_PATH}. Ready for predictions.")

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
    Endpoint for making predictions using the SMOTE-trained model.
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
        message = "Prediction: High Risk (1) - SMOTE Model" if result == 1 else "Prediction: Low Risk (0) - SMOTE Model"
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
    # For local testing
    # app.run(debug=True)
    pass
