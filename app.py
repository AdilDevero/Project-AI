from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import pickle
import numpy as np
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = "data/uploads"
RESULT_FOLDER = "data/results"
MODEL_PATH = "model/model.pkl"
METRICS_PATH = "model/metrics.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load model and metrics
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    model = None
    print("Warning: model.pkl not found. Please run train.py first.")

try:
    with open(METRICS_PATH, "r") as f:
        model_metrics = json.load(f)
    print("Metrics loaded successfully.")
except FileNotFoundError:
    model_metrics = {}
    print("Warning: metrics.json not found.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/model_info", methods=["GET"])
def model_info():
    if not model_metrics:
        return jsonify({"error": "No model metrics available"}), 404
    return jsonify(model_metrics)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    # Save file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Process file and generate bulk predictions
    try:
        data = pd.read_csv(filepath)
        # Ensure only expected features are used
        expected_features = model_metrics.get("feature_names", [])
        if not expected_features:
             return jsonify({"error": "Model features unknown. Train the model first."}), 500
        
        # Check if all expected features are in the CSV
        missing_features = set(expected_features) - set(data.columns)
        if missing_features:
            return jsonify({"error": f"Missing features in CSV: {list(missing_features)}"}), 400
            
        X = data[expected_features]
        predictions = model.predict(X).tolist()
        probabilities = model.predict_proba(X)[:, 1].tolist()
        
        # Save results
        result_df = data.copy()
        result_df["Prediction"] = predictions
        result_df["ChurnProbability"] = probabilities
        
        result_filename = f"predictions_{file.filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        result_df.to_csv(result_path, index=False)
        
        return jsonify({
            "message": "Bulk predictions generated",
            "result_path": result_path,
            "count": len(predictions),
            "filename": result_filename
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

@app.route("/predict_single", methods=["POST"])
def predict_single():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        input_data = request.json
        expected_features = model_metrics.get("feature_names", [])
        
        # Validate all features are present
        missing = [f for f in expected_features if f not in input_data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400
        
        # Prepare input for prediction
        features_values = [float(input_data[f]) for f in expected_features]
        X = np.array(features_values).reshape(1, -1)
        
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
        
        return jsonify({
            "prediction": prediction,
            "churn_probability": round(probability, 4),
            "status": "High Risk" if prediction == 1 else "Healthy"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/download", methods=["GET"])
def download_results():
    result_path = request.args.get("result_path")
    if not result_path or not os.path.exists(result_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(result_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
