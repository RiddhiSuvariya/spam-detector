import pandas as pd
import numpy as np
import re
import string
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

# ✅ Load and clean data
data = pd.read_csv('spam.csv', encoding='latin-1')

# Rename if needed
if 'v1' in data.columns and 'v2' in data.columns:
    data.rename(columns={'v1': 'Category', 'v2': 'Message'}, inplace=True)

# Keep necessary columns
if 'Category' in data.columns and 'Message' in data.columns:
    data = data[['Category', 'Message']]
else:
    raise KeyError("CSV must contain 'Category' and 'Message' columns.")

# Drop duplicates and missing values
data.drop_duplicates(inplace=True)
data.dropna(subset=['Message'], inplace=True)

# Replace labels
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# ✅ Filter only valid categories
data = data[data['Category'].isin(['Spam', 'Not Spam'])]

# ✅ Clean text
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        return text
    return ""

data['Message'] = data['Message'].apply(clean_text)

# ✅ Split data
X = data['Message']
y = data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Vectorize
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ Manual Naive Bayes
word_counts = {'Spam': {}, 'Not Spam': {}}
class_counts = {'Spam': 0, 'Not Spam': 0}

def train_naive_bayes():
    for message, category in zip(X_train, y_train):
        words = message.split()
        class_counts[category] += 1
        for word in words:
            word_counts[category][word] = word_counts[category].get(word, 0) + 1

def predict_naive_bayes(message):
    words = clean_text(message).split()
    total = sum(class_counts.values())
    spam_prob = np.log(class_counts['Spam'] / total)
    not_spam_prob = np.log(class_counts['Not Spam'] / total)
    
    for word in words:
        spam_prob += np.log((word_counts['Spam'].get(word, 0) + 1) / (sum(word_counts['Spam'].values()) + 1))
        not_spam_prob += np.log((word_counts['Not Spam'].get(word, 0) + 1) / (sum(word_counts['Not Spam'].values()) + 1))
    
    return 'Spam' if spam_prob > not_spam_prob else 'Not Spam'

# Train Naive Bayes
train_naive_bayes()

# ✅ Other models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=50, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    accuracies[name] = accuracy_score(y_test, y_pred) * 100

# ✅ Overfitting model
rf_overfit = RandomForestClassifier(n_estimators=1000, max_depth=100, random_state=42)
rf_overfit.fit(X_train_vec, y_train)
y_pred_over = rf_overfit.predict(X_test_vec)
accuracies["Overfitted Random Forest"] = accuracy_score(y_test, y_pred_over) * 100

# ✅ Underfitting model
log_underfit = LogisticRegression(max_iter=100)
log_underfit.fit(X_train_vec, y_train)
y_pred_under = log_underfit.predict(X_test_vec)
accuracies["Underfitted Logistic Regression"] = accuracy_score(y_test, y_pred_under) * 100

# ✅ Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    message = request.form['message']
    prediction = predict_naive_bayes(message)
    return jsonify({'prediction': prediction})

@app.route('/model_accuracies')
def model_accuracies():
    return jsonify(accuracies)

# ✅ Run App
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # use 10000 for Render
    app.run(host='0.0.0.0', port=port)
