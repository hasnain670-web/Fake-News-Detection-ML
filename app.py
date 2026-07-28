import streamlit as st
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ----------------------------------
# Download NLTK Stopwords (Only Once)
# ----------------------------------


try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ----------------------------------
# Load Trained Model
# ----------------------------------

model = joblib.load("models/fake_news_mode.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ----------------------------------
# Text Cleaning Function
# ----------------------------------

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(f"[{string.punctuation}]", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Stemming
    words = [stemmer.stem(word) for word in words]

    # Join words back
    return " ".join(words)

# ----------------------------------
# Sidebar
# ----------------------------------

st.sidebar.title("📰 Fake News Detection")

st.sidebar.info(
    """
### Model
Passive Aggressive Classifier

### NLP
TF-IDF Vectorizer

### Author
Muhammad Hasnain
"""
)

# ----------------------------------
# Main Title
# ----------------------------------

st.title("📰 Fake News Detection")

st.write(
    """
This application predicts whether a news article is **Fake** or **Real**
using Natural Language Processing (NLP) and Machine Learning.
"""
)

st.divider()

# ----------------------------------
# User Input
# ----------------------------------

news = st.text_area(
    "Paste News Article",
    height=250,
    placeholder="Paste the complete news article here..."
)

# ----------------------------------
# Prediction
# ----------------------------------

if st.button("Predict", use_container_width=True):

    if not news.strip():
        st.warning("⚠ Please enter a news article.")
    else:

        cleaned_news = clean_text(news)

        vector = vectorizer.transform([cleaned_news])

        prediction = model.predict(vector)[0]

        st.divider()

        if prediction == 0:
            st.error("🔴 Prediction: Fake News")
        else:
            st.success("🟢 Prediction: Real News")

        st.divider()

        st.subheader("📝 News Article")

        st.write(news)

        st.divider()

        st.subheader("🧹 Processed Text")

        st.write(cleaned_news)

# ----------------------------------
# Footer
# ----------------------------------

st.divider()

st.caption(
    "Built with ❤️ using Streamlit, Scikit-learn, and NLP."
)