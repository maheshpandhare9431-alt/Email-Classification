import joblib

# Load the trained model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to predict spam or ham
def predict_spam(email_text):
    email_vec = vectorizer.transform([email_text])
    prediction = model.predict(email_vec)[0]
    return "Spam" if prediction == 1 else "Ham"

# Take input from terminal
email_input = input("Enter the email content for spam detection:\n")

# Make prediction and print result
result = predict_spam(email_input)
print(f"The email is classified as: {result}")
