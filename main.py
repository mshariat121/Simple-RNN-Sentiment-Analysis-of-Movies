# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words if word]  # skip empty words

    if not encoded_review:
        return None  # empty after filtering

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=300, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def display_confidence_bar(score):
    score = float(score)  # convert from numpy float32
    st.progress(score)
    st.markdown(f"<h4 style='text-align: center;'>Confidence: {score * 100:.2f}%</h4>", unsafe_allow_html=True)

# Streamlit App UI
st.set_page_config(page_title="🎬 IMDB Sentiment Classifier", layout="centered")
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Paste a movie review below, and let the model tell you if it's 👍 Positive or 👎 Negative.")

# User Input
user_input = st.text_area("✍️ Enter your movie review here:")

if st.button("🔍 Classify Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter a valid review.")
    else:
        preprocessed_input = preprocess_text(user_input)

        if preprocessed_input is None:
            st.error("😕 Sorry, the review doesn't contain any recognizable words. Please try again with a longer or clearer review.")
        else:
            prediction = model.predict(preprocessed_input)
            score = prediction[0][0]
            sentiment = '👍 Positive' if score > 0.5 else '👎 Negative'

            # Output
            st.markdown(f"## Sentiment: {sentiment}")
            display_confidence_bar(score if score > 0.5 else 1 - score)

            with st.expander("🧠 View Prediction Score"):
                st.write(f"Raw prediction score: `{score}`")

            with st.expander("☁️ Word Cloud of Your Review"):
                generate_wordcloud(user_input)

else:
    st.info("The app uses a pre-trained RNN to predict if your movie review is positive or negative. Try it!")
