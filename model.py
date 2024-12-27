import joblib

# Load saved model, vectorizer, and label encoder
MODEL_PATH = 'model/model.pkl'
VECTORIZER_PATH = 'model/vectorizer.pkl'
LABEL_ENCODER_PATH = 'model/label_encoder.pkl'

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

def predict_url(url: str) -> str:
    """
    Predict the category of a given URL.
    
    Args:
        url (str): The URL to classify.
    
    Returns:
        str: The predicted category (e.g., 'benign', 'malicious', etc.).
    """
    # Vectorize the input URL
    vect = vectorizer.transform([url])
    
    # Predict the label
    prediction = model.predict(vect)
    
    # Convert numerical label back to category
    category = label_encoder.inverse_transform(prediction)[0]
    return category
