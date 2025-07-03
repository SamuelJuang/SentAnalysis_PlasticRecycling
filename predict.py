import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_model_and_vectorizer():
    model = tf.keras.models.load_model("lstm_model.h5")  
    vectorizer = joblib.load("vectorizer.pkl")
    if not model or not vectorizer:
        raise ValueError("Model or vectorizer not found. Please ensure they are correctly loaded.")
    return model, vectorizer

def uncapitalize(doc):
    return doc.lower()
stemmer = StemmerFactory().create_stemmer()
wpt = nltk.WordPunctTokenizer()
stopword_factory = StopWordRemoverFactory()

stopword = stopword_factory.get_stop_words()


def normalize_document(doc):
    doc = uncapitalize(doc)
    doc = doc.strip()
    doc = wpt.tokenize(doc)
    doc = [stemmer.stem(word) for word in doc]
    doc = [word for word in doc if word not in stopword]
    doc = " ".join(doc)
    return doc

def predict(input_text, model, vectorizer):
    input_text = normalize_document(input_text)
    X_new_seq = vectorizer.texts_to_sequences(input_text)
    X_new_pad = pad_sequences(X_new_seq, padding='post', truncating='post')
    predictions = model.predict(X_new_pad)
    
    return predictions


st.title("Text Prediction App")
st.write("Enter a text to predict the class:")


user_input = st.text_area("Input your text here:")

label_categories = {
    "Recyclability": ["Negative", "Neutral", "Positive"],
    "Recyclability (PET)": ["Negative", "Neutral", "Positive"],
    "Recycling": ["Negative", "Neutral", "Positive"],
    "Future": ["Negative", "Neutral", "Positive"]
}
if st.button("Predict"):
    if user_input:
        model, vectorizer = load_model_and_vectorizer()
        predictions = predict(user_input, model, vectorizer)
        probs = predictions[0]
        label_groups = {
            'Recyclability': ['Recyclability_positive', 'Recyclability_negative', 'Recyclability_neutral'],
            'PET': ['PET_positive', 'PET_negative', 'PET_neutral'],
            'Processing': ['Processing_positive', 'Processing_negative', 'Processing_neutral'],
            'Future': ['Future_positive', 'Future_negative', 'Future_neutral']
        }
        all_labels = sum(label_groups.values(), [])
        label_prob_map = dict(zip(all_labels, probs))

        st.subheader("Predictions:")
        st.write("Text:", user_input)

        for group, labels in label_groups.items():
            group_probs = [label_prob_map[label] for label in labels]
            best_idx = np.argmax(group_probs)
            sentiment = labels[best_idx].split("_")[-1]
            st.write(f"{group}: {sentiment.capitalize()} (Confidence: {group_probs[best_idx]:.2f})")
    else:
        st.error("Please enter some text to predict.")
