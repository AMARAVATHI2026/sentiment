import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv(r"D:\AMVTELURU\projects\Sentiment_Analysis_Project\Sentiment_Analysis_Project\reviews.csv")

print("Dataset Loaded Successfully")
print(df.head())

# -----------------------------------
# Convert Rating to Sentiment
# -----------------------------------
def convert_rating(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

df['sentiment'] = df['Rating'].apply(convert_rating)

# -----------------------------------
# Text Preprocessing
# -----------------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df['clean_review'] = df['ReviewText'].apply(preprocess)

print("\nAfter Preprocessing:")
print(df[['ReviewText', 'clean_review']].head())

# -----------------------------------
# TF-IDF Vectorization
# -----------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_review'])
y = df['sentiment']

print("\nTF-IDF Shape:", X.shape)

# -----------------------------------
# Train Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# Model Training
# -----------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------------
# Evaluation
# -----------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------------
# Confusion Matrix Plot
# -----------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------------
# Custom Review Test
# -----------------------------------
print("\n--- Test Custom Review ---")
while True:
    review = input("Enter a review (or type 'exit'): ")
    if review.lower() == "exit":
        break
    review_clean = preprocess(review)
    review_vector = vectorizer.transform([review_clean])
    prediction = model.predict(review_vector)
    print("Predicted Sentiment:", prediction[0])