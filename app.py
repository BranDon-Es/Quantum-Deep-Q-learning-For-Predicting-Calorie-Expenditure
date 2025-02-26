import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the meta-model (Linear Regression trained on base models' predictions)
meta_model = joblib.load("Best_Ensemble_Model.pkl")

print(f"Meta-model type: {type(meta_model)}")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the JSON request data
        data = request.get_json()

        # Check if necessary fields are provided
        if "features" not in data or "base_model_predictions" not in data:
            return jsonify({"error": "Missing 'features' or 'base_model_predictions' in the input data"}), 400

        # Prepare the features and base model predictions
        features = np.array(data["features"]).reshape(1, -1)
        model_predictions = np.array([data["base_model_predictions"]])

        # Ensure that the number of predictions matches what the model expects
        if model_predictions.shape[1] != 11:  # Based on your model's input size (e.g., 11 features)
            return jsonify({"error": "Expected 11 base model predictions, but got a different number."}), 400

        # Get the final prediction using the meta-model
        final_prediction = meta_model.predict(model_predictions)

        # Return the prediction as a JSON response
        return jsonify({"calories_burned": final_prediction.tolist()})

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
