# fraud_email_detection.py

# =======================
# 1. Import Libraries
# =======================
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# =======================
# 2. Load Dataset
# =======================
# Replace with your dataset path
data = pd.read_csv("dataset/fraud_email.csv")  

texts = data['Text']     # column containing email text
labels = data['Class']   # column containing labels (fraud=1, non-fraud=0)

# =======================
# 3. Preprocessing
# =======================
def preprocess(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.split()
    else:
        return []

# Apply preprocessing + stopword removal
preprocessed_texts = texts.apply(lambda x: [word for word in preprocess(x) if word not in stop_words])
processed_texts = preprocessed_texts.apply(lambda x: ' '.join(x))

# =======================
# 4. Feature Extraction
# =======================
vectorizer = TfidfVectorizer(max_df=0.7, min_df=5, ngram_range=(1, 2), max_features=10000)
X = vectorizer.fit_transform(processed_texts)

# Feature selection
selector = SelectKBest(chi2, k=5000)
X_selected = selector.fit_transform(X, labels)

# Handle class imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_selected, labels)

# =======================
# 5. Train/Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# =======================
# 6. Model Training
# =======================
classifier = LogisticRegression(max_iter=1000, class_weight='balanced')

param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}

grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_classifier = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# =======================
# 7. Model Evaluation
# =======================
y_pred = best_classifier.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

cross_val_scores = cross_val_score(best_classifier, X_resampled, y_resampled, cv=5)
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean CV Accuracy: {np.mean(cross_val_scores):.4f}")

# =======================
# 8. Prediction Function
# =======================
def predict_fraud(text, model, vectorizer, selector):
    preprocessed_text = ' '.join([word for word in preprocess(text) if word not in stop_words])
    X_input = vectorizer.transform([preprocessed_text])
    X_input_selected = selector.transform(X_input)
    fraud_proba = model.predict_proba(X_input_selected)[0][1]
    return fraud_proba * 100

# =======================
# 9. Example Predictions
# =======================
examples = [
    """With all due respect, I want you to read my letter with one mind and help me...""",
    """I hope you are well when you get this email. My name is [Name], and I'm the son of the late [Famous Person]...""",
    """Dear: Account Owner, Our records indicate that you are enrolled in the University of California paperless W2 Program...""",
    """Hello, We've noticed that some of your account information appears to be missing or incorrect...""",
    """From: BankOfAmerica Subject: Irregular Activity Date: 10/20/2016..."""
]

for i, text in enumerate(examples, 1):
    fraud_likelihood = predict_fraud(text, best_classifier, vectorizer, selector)
    print(f"\nExample {i} Fraud Likelihood: {fraud_likelihood:.2f}%")
