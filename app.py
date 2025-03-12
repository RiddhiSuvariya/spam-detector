
#----------Working---
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from imblearn.over_sampling import RandomOverSampler
# from flask import Flask, request, render_template, jsonify

# # Load data
# data = pd.read_csv(r'D:\spam\spam.csv')

# # Data Preprocessing
# data.drop_duplicates(inplace=True)
# data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# # Splitting into messages and categories
# mess = data['Message']
# cat = data['Category']

# # Train-test split (80% training, 20% testing)
# mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2, random_state=42)

# # Text Vectorization
# cv = CountVectorizer(stop_words='english')
# features_train = cv.fit_transform(mess_train)
# features_test = cv.transform(mess_test)

# # Handle class imbalance using Random OverSampling
# ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
# features_train_resampled, cat_train_resampled = ros.fit_resample(features_train, cat_train)

# # List of models
# models = {
#     "Naïve Bayes": MultinomialNB(),
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "Support Vector Machine": SVC(kernel='linear'),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
# }

# # Train and evaluate models
# for name, model in models.items():
#     model.fit(features_train_resampled, cat_train_resampled)
#     predictions = model.predict(features_test)
#     accuracy = accuracy_score(cat_test, predictions)
#     print(f"{name} Accuracy: {accuracy:.4f}")
#     print(classification_report(cat_test, predictions))
#     print("=" * 60)

# # Prediction function
# def predict_message(model, message):
#     input_message = cv.transform([message])
#     prediction = model.predict(input_message)[0]
#     return prediction  # Returns "Spam" or "Not Spam"

# # Flask application
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')  # Render the home page

# @app.route('/predict', methods=['POST'])
# def predict_route():
#     message = request.form['message']
#     chosen_model = models["Naïve Bayes"]  # Default model for prediction
#     prediction = predict_message(chosen_model, message)
#     return jsonify({'prediction': prediction})


# @app.route('/model_accuracies')
# def model_accuracies():
#     accuracies = {
#         "Naïve Bayes": 96.12,
#         "Logistic Regression": 98.26,
#         "Support Vector Machine": 98.45,
#         "Random Forest": 98.16,
#         "K-Nearest Neighbors": 94.48
#     }
#     return {"accuracies": accuracies}

# if __name__ == '__main__':
#     app.run(debug=True)


#trying-- menual--
# import pandas as pd
# import numpy as np
# import re
# import string
# from sklearn.linear_model import LogisticRegression
# from flask import Flask, request, render_template, jsonify

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC  # Support Vector Machine
# from sklearn.ensemble import RandomForestClassifier  # Random Forest
# from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Load data
# data = pd.read_csv(r'D:\spam\spam.csv')
# data
# # Data Preprocessing
# data.drop_duplicates(inplace=True)
# data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(f"[{string.punctuation}]", "", text)
#     return text

# data['Message'] = data['Message'].apply(clean_text)

# # Splitting data
# train_size = int(0.8 * len(data))
# train_data = data[:train_size]
# test_data = data[train_size:]

# # Count words in each class
# word_counts = {'Spam': {}, 'Not Spam': {}}
# class_counts = {'Spam': 0, 'Not Spam': 0}

# def train_naive_bayes():
#     for _, row in train_data.iterrows():
#         category = row['Category']
#         words = row['Message'].split()
#         class_counts[category] += 1
#         for word in words:
#             if word not in word_counts[category]:
#                 word_counts[category][word] = 1
#             else:
#                 word_counts[category][word] += 1

# train_naive_bayes()

# def predict(message):
#     words = clean_text(message).split()
#     spam_prob = np.log(class_counts['Spam'] / sum(class_counts.values()))
#     not_spam_prob = np.log(class_counts['Not Spam'] / sum(class_counts.values()))
    
#     for word in words:
#         spam_prob += np.log((word_counts['Spam'].get(word, 0) + 1) / (sum(word_counts['Spam'].values()) + 1))
#         not_spam_prob += np.log((word_counts['Not Spam'].get(word, 0) + 1) / (sum(word_counts['Not Spam'].values()) + 1))
    
#     return 'Spam' if spam_prob > not_spam_prob else 'Not Spam'

# # Flask application
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict_route():
#     message = request.form['message']
#     prediction = predict(message)
#     return jsonify({'prediction': prediction})

# @app.route('/model_accuracies')
# def model_accuracies():
#     accuracies = {
#         "Naïve Bayes": 96.12,
#         "Logistic Regression": 98.26,
#         "Support Vector Machine": 98.45,
#         "Random Forest": 98.16,
#          "K-Nearest Neighbors": 94.48
#     }
#     return jsonify(accuracies)

# if __name__ == '__main__':
#     app.run(debug=True)

#----

import pandas as pd
import numpy as np
import re
import string
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Data
data = pd.read_csv('spam.csv')
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

data['Message'] = data['Message'].apply(clean_text)

# Split Data
X = data['Message']
y = data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Naïve Bayes Manual Implementation
word_counts = {'Spam': {}, 'Not Spam': {}}
class_counts = {'Spam': 0, 'Not Spam': 0}

def train_naive_bayes():
    for message, category in zip(X_train, y_train):
        words = message.split()
        class_counts[category] += 1
        for word in words:
            if word not in word_counts[category]:
                word_counts[category][word] = 1
            else:
                word_counts[category][word] += 1

train_naive_bayes()

def predict_naive_bayes(message):
    words = clean_text(message).split()
    spam_prob = np.log(class_counts['Spam'] / sum(class_counts.values()))
    not_spam_prob = np.log(class_counts['Not Spam'] / sum(class_counts.values()))
    
    for word in words:
        spam_prob += np.log((word_counts['Spam'].get(word, 0) + 1) / (sum(word_counts['Spam'].values()) + 1))
        not_spam_prob += np.log((word_counts['Not Spam'].get(word, 0) + 1) / (sum(word_counts['Not Spam'].values()) + 1))
    
    return 'Spam' if spam_prob > not_spam_prob else 'Not Spam'

# Train Models
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

# Overfitting: Deep Random Forest
rf_overfit = RandomForestClassifier(n_estimators=1000, max_depth=100, random_state=42)
rf_overfit.fit(X_train_vec, y_train)
y_pred_rf_overfit = rf_overfit.predict(X_test_vec)
accuracies["Overfitted Random Forest"] = accuracy_score(y_test, y_pred_rf_overfit) * 100

# Underfitting: Basic Logistic Regression without tuning
log_reg_underfit = LogisticRegression(max_iter=100)
log_reg_underfit.fit(X_train_vec, y_train)
y_pred_log_underfit = log_reg_underfit.predict(X_test_vec)
accuracies["Underfitted Logistic Regression"] = accuracy_score(y_test, y_pred_log_underfit) * 100

# Flask application
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

if __name__ == '__main__':
    app.run(debug=True)


import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

