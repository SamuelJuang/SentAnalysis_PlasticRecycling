import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import emoji

@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model('lstm_model.keras')
    tokenizer = joblib.load("tokenizer.pkl")
    if not model or not tokenizer:
        raise ValueError("Model or tokenizer not found. Please ensure they are correctly loaded.")
    return model, tokenizer

def uncapitalize(doc):
    return doc.lower()
stemmer = StemmerFactory().create_stemmer()
wpt = nltk.WordPunctTokenizer()
stopword_factory = StopWordRemoverFactory()

stopword = stopword_factory.get_stop_words()

def clean_tweet_id(text):
    text = text.lower()
    text = emoji.demojize(text)
    text = re.sub(r':([a-zA-Z0-9_]+):', r'\1', text)
    text = text.replace('_', ' ')
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    factory = StopWordRemoverFactory()
    stopwords = set(factory.get_stop_words())
    text = ' '.join([word for word in text.split() if word not in stopwords])
    
    return text

def normalize_document(doc):
    doc = uncapitalize(doc)
    doc = doc.strip()
    doc = wpt.tokenize(doc)
    doc = [stemmer.stem(word) for word in doc]
    doc = [word for word in doc if word not in stopword]
    doc = " ".join(doc)
    return doc

def predict(input_text, model, tokenizer):
    input_text = clean_tweet_id(input_text)
    input_text = normalize_document(input_text)
    X_new_seq = tokenizer.texts_to_sequences(input_text)
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
        model, tokenizer = load_model_and_tokenizer()
        predictions = predict(user_input, model, tokenizer)
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
