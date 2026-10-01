import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Page configuration
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="centered"
)

# Load model
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

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))


# Text preprocessing
def preprocess_text(text):
    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = "".join(
        char for char in text if not char.isdigit()
    )

    text = "".join(
        char for char in text if char.isascii()
    )

    words = word_tokenize(text)

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Header
st.title("😊 Emotion Classifier")
st.write(
    "An NLP-based machine learning application "
    "that predicts the emotion expressed in a sentence."
)

st.divider()

# About section
with st.expander("ℹ️ About this project"):
    st.write(
        "This project uses Natural Language Processing (NLP) "
        "and a Linear Support Vector Machine (Linear SVM) model "
        "to classify text into different emotion categories."
    )

    st.write("**Feature Extraction:** TF-IDF")
    st.write("**Machine Learning Model:** Linear SVM")
    st.write("**Test Accuracy:** 90.16%")


# Input section
st.subheader("📝 Enter your text")

text = st.text_area(
    "Type a sentence below:",
    placeholder="Example: I am very happy today!",
    height=150
)


# Prediction
if st.button("🔮 Predict Emotion", use_container_width=True):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:
        cleaned_text = preprocess_text(text)

        text_vector = vectorizer.transform(
            [cleaned_text]
        )

        prediction = model.predict(text_vector)[0]

        emotion = emotion_mapping[prediction]

        st.divider()

        st.subheader("🎯 Prediction")

        st.success(
            f"Predicted Emotion: **{emotion.upper()}**"
        )


# Example sentences
st.divider()

st.subheader("💡 Try an example")

examples = [
    "I am extremely happy today!",
    "I feel very sad and lonely.",
    "I am really angry right now.",
    "I am scared about what will happen."
]

for example in examples:
    st.write(f"• {example}")


# Footer
st.divider()

st.caption(
    "Built using Python, NLP, TF-IDF and Linear SVM"
)
