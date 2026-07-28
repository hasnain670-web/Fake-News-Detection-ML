import streamlit as  st
import joblib 
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download("stopwords")
stop_words=set(stopwords.words("english"))
stemmer=PorterStemmer()

model=joblib.load("models/fake_news_mode.pkl")
vectorizer=joblib.load("models/vectorizer.pkl")

def clean_text(text):
    text=text.lower()
    text=re.sub(f"[{string.punctuation}]","",text)
    text=re.sub(r"\d+","",text)
    text=re.sub(r"\s+","",text)
    words=text.split()
    words=[word for word in  words if word not in stop_words ]
    words=[stemmer.stem(word) for word in words]
    return "".join(words)

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.sidebar.title("📰 Fake News Detection")
st.sidebar.info(
    """
    Model:
    Passive Aggressive Classifier
    
    NLP:
    TF-IDF Vectorizer

    Author:
    Muhammad Hasnain
  """
)
st.title("📰 Fake News Detection")
st.write(
    "Enter a news article below to check whether it is **Fake** or **Real**."
)
st.divider()

news=st.text_area(
    "Paste News Article",
    height=250
)

if st.button("Predict",use_container_width=True):
    if news.strip() == "":
        st.warning("Please enter some news.")
    else:

        cleaned = clean_text(news)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)

        st.divider()

        if prediction[0] == 0:
            st.error("🔴 Fake News")
        else:
            st.success("🟢 Real News")

        st.divider()

        st.subheader("News Entered")

        st.write(news)
