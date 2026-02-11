# Import Dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load Dataset
# Download 'spam.csv' from: https://www.kaggle.com/uciml/sms-spam-collection-dataset
data = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']
data = data.dropna(subset=['label', 'message'])  # Removes missing data, if any

# Preprocessing: Encode labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Features and targets
X = data['message']
y = data['label']

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization: Convert text to numeric features
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Model Training
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Prediction & Evaluation
y_pred = model.predict(X_test_counts)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict function for new emails
def predict_spam(email):
    email_counts = vectorizer.transform([email])
    prediction = model.predict(email_counts)[0]
    return "Spam" if prediction == 1 else "Ham"

# Examples
print(predict_spam("Congratulations! You've won a $1000 gift card. Call now!"))
print(predict_spam("Hi, are we meeting tomorrow at 10am?"))

# Save trained model and vectorizer for reuse (e.g., Streamlit, Flask)
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
