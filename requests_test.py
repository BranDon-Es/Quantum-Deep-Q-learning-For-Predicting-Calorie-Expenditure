import requests
import json

url = "http://127.0.0.1:5000/predict"
headers = {"Content-Type": "application/json"}

data = {
    "features": [25, 70, 175, 80, 30],  # Example feature values
    "base_model_predictions": [150, 160, 155, 162, 158, 148, 157, 159, 154, 151, 149]  # Example predictions from all 11 models
}


response = requests.post(url, json=data, headers=headers)

# Check if the response is empty or not
if response.status_code == 200:
    try:
        print("Response:", response.json())
    except ValueError:
        print("Response not in JSON format. Raw response text:", response.text)
else:
    print(f"Request failed with status code {response.status_code}.")
