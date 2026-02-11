from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def predict_spam(email_text):
    email_vec = vectorizer.transform([email_text])
    prediction = model.predict(email_vec)[0]
    return "Spam" if prediction == 1 else "Ham"

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    email_text = ""

    if request.method == 'POST':
        email_text = request.form['email']
        prediction = predict_spam(email_text)

    return render_template('index.html', prediction=prediction, email_text=email_text)

if __name__ == '__main__':
    app.run(debug=True)
