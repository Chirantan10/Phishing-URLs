import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the model, vectorizer, and label encoder
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')  # Ensure you save the LabelEncoder when training the model

def predict_url_type(url):
    """
    Predicts the type of a given URL using the trained model.
    """
    # Vectorize the input URL
    url_vect = vectorizer.transform([url])

    # Predict using the trained model
    prediction = model.predict(url_vect)

    # Convert the numerical label back to the original category
    predicted_label = label_encoder.inverse_transform(prediction)

    return predicted_label[0]

# Routes
@app.route('/')
def home():
    return "Welcome to the URL Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict the type of a URL.
    Expects JSON input: {"url": "<URL>"}
    """
    try:
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        # Predict the URL type
        predicted_type = predict_url_type(url)

        return jsonify({"url": url, "prediction": predicted_type})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get the port from the environment variable or use 5000 as a fallback
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
