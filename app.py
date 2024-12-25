from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')  # assuming you saved the vectorizer

@app.route('/')
def home():
    return "Welcome to the URL Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the URL from the request
        data = request.get_json()
        url = data.get('url')

        # Vectorize the URL using the pre-trained vectorizer
        url_vect = vectorizer.transform([url])

        # Make the prediction using the trained model
        prediction = model.predict(url_vect)[0]

        # Return the result
        return jsonify({'prediction': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
