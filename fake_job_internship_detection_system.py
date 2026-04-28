# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.sparse import hstack

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# ================================
# 2. LOAD DATA
# ================================
df = pd.read_csv('fake_job_postings.csv')

# ================================
# 3. CLEANING
# ================================
df = df.fillna('')
df = df.drop_duplicates()

# Select important columns
df = df[['title', 'description', 'salary_range', 'fraudulent']]

# ================================
# 4. TEXT PROCESSING (MATCH APP)
# ================================
# Combine title + description (IMPORTANT)
df['text'] = (df['title'] + " " + df['description']).str.lower()

# Remove punctuation (same as app)
df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['text'] = df['text'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

# ================================
# 5. FEATURE ENGINEERING
# ================================
df['salary_flag'] = df['salary_range'].apply(lambda x: 0 if x == '' else 1)
df['desc_length'] = df['text'].apply(len)

y = df['fraudulent']

# ================================
# 6. TF-IDF
# ================================
tfidf = TfidfVectorizer(max_features=5000)
X_text = tfidf.fit_transform(df['text'])

X = hstack((X_text, df[['salary_flag', 'desc_length']].values))

# ================================
# 7. TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 8. MODEL TRAINING
# ================================
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# ================================
# 9. EVALUATION
# ================================
y_pred = rf.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ================================
# 10. IMPORTANT VISUALIZATIONS
# ================================

# Fake vs Real
sns.countplot(x=y)
plt.title("Fake vs Real Job Distribution")
plt.show()

# Description length
sns.boxplot(x=y, y=df['desc_length'])
plt.title("Description Length Comparison")
plt.show()

# Salary vs Fraud
sns.countplot(x='salary_flag', hue='fraudulent', data=df)
plt.title("Salary Presence vs Fraud")
plt.show()

# ================================
# 11. SAVE MODEL
# ================================
pickle.dump(rf, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

print("✅ Model saved successfully!")