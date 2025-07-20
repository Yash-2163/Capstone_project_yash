import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- IMPORTANT ---
# This backend requires your custom preprocessing module to load the model.
# Ensure 'preprocessing.py' is in the same directory as this 'app.py' file.
try:
    from preprocessing import CategoricalPreprocessor  # Importing custom preprocessing transformer
    print("[INFO] Custom 'preprocessing' module loaded.")
except ImportError:
    # If the file is missing, log the error. You may want to exit in production.
    print("[ERROR] 'preprocessing.py' not found. Model loading will fail.")
    # sys.exit(1) 

# --- Flask App Initialization ---
app = Flask(__name__)  # Creating Flask application instance
CORS(app)  # Enable Cross-Origin Resource Sharing (useful if frontend is hosted elsewhere)

# --- Load the Trained Model ---
# Assumes the 'final_model' directory is at the project root, one level above this 'backend' directory.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Parent directory of the current file
MODEL_PATH = os.path.join(BASE_DIR, "final_model", "final_model_pipeline.pkl")  # Path to trained model file

model = None  # Model will be loaded here
try:
    model = joblib.load(MODEL_PATH)  # Load model using joblib
    print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    # Handle missing model file
    print(f"[ERROR] Model file not found at {MODEL_PATH}. Please ensure the training script has been run and the model is saved.")
except Exception as e:
    # Handle any other model loading errors
    print(f"[ERROR] An error occurred while loading the model: {e}")

# --- API Endpoints ---

@app.route("/predict", methods=["POST"])
def predict():
    """Receives a file upload, runs prediction, and returns results."""
    if model is None:
        # Model not loaded properly
        return jsonify({"error": "Model is not loaded. Check server logs."}), 500

    if 'file' not in request.files:
        # No file part in the form
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        # File field is empty
        return jsonify({"error": "No file selected"}), 400

    try:
        input_df = pd.read_csv(file)  # Read uploaded CSV as a DataFrame

        # The loaded model is a full pipeline, including preprocessing and prediction
        predictions = model.predict(input_df)  # Class labels
        probabilities = model.predict_proba(input_df)[:, 1]  # Probability of class 1 (converted)

        # Add predictions to the original data
        results_df = input_df.copy()
        results_df['Predicted_Conversion'] = predictions
        results_df['Prediction_Probability'] = [f"{p:.2%}" for p in probabilities]  # Format as percentage

        # Return the results in JSON format
        return jsonify({"predictions": results_df.to_dict(orient='records')})

    except Exception as e:
        # Handle any error during prediction
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """A simple endpoint to check if the service is running and the model is loaded."""
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == "__main__":
    # Runs the Flask app. Port 5001 chosen (you can change as needed).
    app.run(host="0.0.0.0", port=5001, debug=True)
