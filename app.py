
import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="centered"
)


# -----------------------------
# Load saved model
# -----------------------------

@st.cache_resource
def load_model():

    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("emotion_mapping.pkl", "rb") as f:
        emotion_mapping = pickle.load(f)

    return model, vectorizer, emotion_mapping


model, vectorizer, emotion_mapping = load_model()


# -----------------------------
# NLTK
# -----------------------------

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))


# -----------------------------
# Text preprocessing
# -----------------------------

def preprocess_text(text):

    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = "".join(
        char for char in text
        if not char.isdigit()
    )

    text = "".join(
        char for char in text
        if char.isascii()
    )

    words = word_tokenize(text)

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# -----------------------------
# UI
# -----------------------------

st.title("😊 Emotion Classifier")

st.write(
    "Enter a sentence and the NLP model will predict its emotion."
)

text = st.text_area(
    "Enter your text:",
    placeholder="Example: I am very happy today!",
    height=150
)


if st.button("Predict Emotion"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        cleaned_text = preprocess_text(text)

        text_vector = vectorizer.transform([cleaned_text])

        prediction = model.predict(text_vector)[0]

        emotion = emotion_mapping[prediction]

        st.success(
            f"Predicted Emotion: **{emotion.upper()}**"
        )
