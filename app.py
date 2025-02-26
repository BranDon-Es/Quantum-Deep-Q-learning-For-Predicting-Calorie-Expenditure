import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
meta_model = joblib.load("Best_Ensemble_Model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received Data:", data)  # Debugging: Print received JSON data

        # Validate input
        if "features" not in data or "base_model_predictions" not in data:
            raise ValueError("Invalid input: Missing required fields")

        features = np.array(data["features"]).reshape(1, -1)
        base_model_predictions = np.array(data["base_model_predictions"]).reshape(1, -1)

        # Combine user features + base model predictions
        full_features = np.concatenate((features, base_model_predictions), axis=1)

        # Ensure no missing values
        if np.isnan(full_features).any():
            raise ValueError("Invalid input: Contains NaN or missing values")

        # Get final prediction
        final_prediction = meta_model.predict(full_features)

        return jsonify({"calories_burned": final_prediction.tolist()})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400  # Return error with HTTP 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
