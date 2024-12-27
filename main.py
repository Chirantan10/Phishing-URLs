import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv('dataset/malicious_phish.csv')

# Step 1: Split the dataset into features (X) and target (y)
X = data['url']
y = data['type']

# Step 2: Convert categorical target values into numerical labels using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 4: Vectorize the URLs using TF-IDF (character-level n-grams)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Function to fit model, predict, and evaluate
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Display accuracy score
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Display classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Display confusion matrix
    print("Confusion Matrix:",confusion_matrix(y_test, y_pred))

# Logistic Regression
print("Logistic Regression:")
log_reg = LogisticRegression()
evaluate_model(log_reg, X_train_vect, y_train, X_test_vect, y_test)

import joblib
# Save the trained model and vectorizer
joblib.dump(log_reg , 'models/log_reg.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
# Load the pre-trained model and vectorizer
# model = joblib.load('model.pkl')
# vectorizer = joblib.load('vectorizer.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
