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
        
        recycle_prediction = np.argmax(predictions[0], axis=1)
        pet_prediction = np.argmax(predictions[1], axis=1)
        process_prediction = np.argmax(predictions[2], axis=1)
        future_prediction = np.argmax(predictions[3], axis=1)
        st.subheader("Predictions:")
        st.write("Predicted Text:", user_input)
        st.write("Recyclability Prediction:", label_categories['Recyclability'][recycle_prediction[0]])
        st.write("Recyclability (PET) Prediction:", label_categories['Recyclability (PET)'][pet_prediction[0]])
        st.write("Recycling Prediction:", label_categories['Recycling'][process_prediction[0]])
        st.write("Future Prediction:", label_categories['Future'][future_prediction[0]])
    else:
        st.error("Please enter some text to predict.")
