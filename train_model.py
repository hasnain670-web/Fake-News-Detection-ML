import pandas as pd
import re
import string
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -------------------------------------------------------
# Download NLTK Stopwords
# -------------------------------------------------------

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true], ignore_index=True)

# Shuffle Dataset

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(df.head())
print(df.shape)

# -------------------------------------------------------
# Visualization
# -------------------------------------------------------

sns.countplot(x="label", data=df)
plt.title("Fake vs Real News")
plt.show()

# -------------------------------------------------------
# Text Cleaning Function
# -------------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(f"[{string.punctuation}]", " ", text)

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)

# -------------------------------------------------------
# Feature Engineering
# -------------------------------------------------------

df["content"] = df["title"] + " " + df["text"]

df["content"] = df["content"].apply(clean_text)

# Save cleaned dataset

df.to_csv("dataset/news_clean.csv", index=False)

X = df["content"]
y = df["label"]

# -------------------------------------------------------
# TF-IDF Vectorizer
# -------------------------------------------------------

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

# -------------------------------------------------------
# Train Test Split
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -------------------------------------------------------
# Models
# -------------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(),

    "Naive Bayes": MultinomialNB(),

    "Passive Aggressive": PassiveAggressiveClassifier(
        max_iter=1000,
        random_state=42
    )

}

best_model = None
best_name = ""
best_accuracy = 0

print("\n========== MODEL RESULTS ==========\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(y_test, prediction)

    recall = recall_score(y_test, prediction)

    f1 = f1_score(y_test, prediction)

    print("=" * 60)

    print(name)

    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report\n")

    print(classification_report(y_test, prediction))

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_name = name

        best_prediction = prediction

# -------------------------------------------------------
# Best Model
# -------------------------------------------------------

print("\n===============================")

print("Best Model :", best_name)

print("Accuracy   :", best_accuracy)

print("===============================\n")

# -------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------

cm = confusion_matrix(y_test, best_prediction)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake", "Real"]
)

disp.plot(cmap="Blues")

plt.title(f"{best_name} Confusion Matrix")

plt.show()

# -------------------------------------------------------
# Save Model
# -------------------------------------------------------

joblib.dump(
    best_model,
    "models/fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)

print("\nModel Saved Successfully!")

print("Vectorizer Saved Successfully!")