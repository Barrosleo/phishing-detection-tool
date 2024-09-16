pip install pandas scikit-learn nltk email

import nltk
nltk.download('punkt')
nltk.download('stopwords')

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from email import policy
from email.parser import BytesParser
import os

# Load and Preprocess the Dataset
def load_data(directory):
    emails = []
    labels = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            with open(os.path.join(root, file), 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)
                emails.append(msg.get_payload())
                labels.append(1 if 'phishing' in root else 0)
    return emails, labels

emails, labels = load_data('path_to_email_dataset')

# Feature Extraction
stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=stop_words)

X = vectorizer.fit_transform(emails)
y = pd.Series(labels)

# Train the Machine Learning Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Detect Phishing Emails
def detect_phishing(email_content):
    email_vector = vectorizer.transform([email_content])
    prediction = model.predict(email_vector)
    return 'Phishing' if prediction[0] == 1 else 'Not Phishing'

# Example usage
email_content = "Your email content here"
print(detect_phishing(email_content))

