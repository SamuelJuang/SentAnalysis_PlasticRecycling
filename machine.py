from sklearn.model_selection import train_test_split
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
import joblib
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


class StreamlitProgressBarCallback(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.current_epoch = 0
        self.logs = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1
        progress = int((self.current_epoch) / self.epochs * 100)
        self.progress_bar.progress(progress)
        self.status_text.text(f"Epoch {self.current_epoch}/{self.epochs} completed.")

def lstm(lstm_units, dense1_units, dropout1_rate,
          dense2_units, dropout2_rate, dense3_units, dropout3_rate, epochs, batch_size, bidirectional=True):
    if(os.path.exists('oversampled_data.csv')):
        df_encoded = pd.read_csv('oversampled_data.csv')
    else:
        df_encoded = pd.read_csv('cleaned_and_normalized_data.csv')
    status = st.empty()
    bar = st.progress(0)

    # TRAIN TEST SPLIT
    status.text("Splitting the data into train and test sets...")
    bar.progress(0.1)
    y_recycle = df_encoded[[ 
        "Recycling_Negative", "Recycling_Neutral", "Recycling_Positive"
    ]]
    y_pet = df_encoded[[ 
        "Recyclability (PET)_Negative", "Recyclability (PET)_Neutral", "Recyclability (PET)_Positive"
    ]]
    y_process = df_encoded[[ 
        "Recyclability_Negative", "Recyclability_Neutral", "Recyclability_Positive"
    ]]
    y_effective = df_encoded[[ 
        "Future_Negative", "Future_Neutral", "Future_Positive"
    ]]

    # Features
    X = df_encoded["text"]  # Or other input features

    # Train-test split for multiple labels
    X_train, X_test, y_recycle_train, y_recycle_test, y_pet_train, y_pet_test, \
    y_process_train, y_process_test, y_effective_train, y_effective_test = train_test_split(
        X, y_recycle, y_pet, y_process, y_effective, test_size=0.2, random_state=42
    )
    st.session_state.y_effective_test = y_effective_test
    st.session_state.y_process_test = y_process_test
    st.session_state.y_pet_test = y_pet_test
    st.session_state.y_recycle_test = y_recycle_test

    status.text("Tokenizing the data...")
    bar.progress(0.2)
    vocab_size = 10000 
    max_len = 100  

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
    st.session_state.X_test_pad = X_test_pad
    
    # Save the vectorizer to a file
    joblib.dump(tokenizer, 'vectorizer.pkl')
   

    label_columns = [
        "Recyclability_Negative", "Recyclability_Neutral", "Recyclability_Positive",
        "Recyclability (PET)_Negative", "Recyclability (PET)_Neutral", "Recyclability (PET)_Positive",
        "Recycling_Negative", "Recycling_Neutral", "Recycling_Positive",
        "Future_Negative", "Future_Neutral", "Future_Positive"
    ]
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100
    max_len = X_train_pad.shape[1]

    (lstm_units, dense1_units, dropout1_rate,
          dense2_units, dropout2_rate, dense3_units, dropout3_rate, epochs, batch_size)

    bar.progress(0.5)
    inputs = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(inputs)
    if(bidirectional):
        x = Bidirectional(LSTM(lstm_units))(x)
    else:
        x = LSTM(lstm_units)(x)
    x = Dense(dense1_units, activation='relu')(x)
    x = Dropout(dropout1_rate)(x)
    x = Dense(dense2_units, activation='relu')(x)
    x = Dropout(dropout2_rate)(x)
    x = Dense(dense3_units, activation='relu')(x)
    x = Dropout(dropout3_rate)(x)

    output_general = Dense(3, activation='softmax', name='recycle')(x)
    output_pet = Dense(3, activation='softmax', name='pet')(x)
    output_process = Dense(3, activation='softmax', name='process')(x)
    output_future = Dense(3, activation='softmax', name='future')(x)

    status.text("training the model...")
    model = Model(inputs=inputs, outputs=[output_general, output_pet, output_process, output_future])
    model.compile(
        optimizer='adam',
        loss={
            'recycle': 'categorical_crossentropy',
            'pet': 'categorical_crossentropy',
            'process': 'categorical_crossentropy',
            'future': 'categorical_crossentropy'
        },
        metrics={
            'recycle': 'accuracy',
            'pet': 'accuracy',
            'process': 'accuracy',
            'future': 'accuracy'
        }
    )
    bar.progress(0.8)
    # ---- TRAIN ----
    callback = StreamlitProgressBarCallback(10)
    history = model.fit(
        X_train_pad,
        {
            'recycle': y_recycle_train,
            'pet': y_pet_train,
            'process': y_process_train,
            'future': y_effective_train
        },
        validation_data=(
            X_test_pad,
            {
                'recycle': y_recycle_test,
                'pet': y_pet_test,
                'process': y_process_test,
                'future': y_effective_test
            }
        ),
        epochs=10,
        batch_size=32,
        callbacks=[callback, EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )

    bar.progress(0.9)
    status.success("Model training complete!")

   
    bar.progress(1.0)
    
    #Show training history
    st.subheader("Total Loss per Epoch")
    fig_loss = plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Total Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig_loss)

    # Outputs to visualize
    outputs = ['recycle', 'pet', 'process', 'future']

    # Accuracy plots per output
    st.subheader("Accuracy per Epoch for Each Output")
    for output in outputs:
        fig_acc = plt.figure(figsize=(8, 5))
        plt.plot(history.history[f'{output}_accuracy'], label=f'{output.capitalize()} Train Accuracy')
        plt.plot(history.history[f'val_{output}_accuracy'], label=f'{output.capitalize()} Val Accuracy')
        plt.title(f'{output.capitalize()} Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig_acc)

    # Loss plots per output
    st.subheader("Loss per Epoch for Each Output")
    for output in outputs:
        fig_loss = plt.figure(figsize=(8, 5))
        plt.plot(history.history[f'{output}_loss'], label=f'{output.capitalize()} Train Loss')
        plt.plot(history.history[f'val_{output}_loss'], label=f'{output.capitalize()} Val Loss')
        plt.title(f'{output.capitalize()} Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig_loss)
    
   
    # Save the model
    model.save('lstm_model.h5')

    

st.title("Multi-Label LSTM Training")

if os.path.exists('cleaned_and_normalized_data.csv') or os.path.exists('oversampled_data.csv'):

    st.title("Multi-Label LSTM Training")

    lstm_units = st.slider("LSTM Units", 32, 256, 128, step=16)
    bidirectional = st.checkbox("Use Bidirectional LSTM", value=True)
    dense1_units = st.slider("Dense Layer 1 Units", 64, 1024, 512, step=64)
    dropout1_rate = st.slider("Dropout Rate after Dense 1", 0.0, 0.7, 0.5, step=0.05)

    dense2_units = st.slider("Dense Layer 2 Units", 64, 512, 256, step=32)
    dropout2_rate = st.slider("Dropout Rate after Dense 2", 0.0, 0.5, 0.3, step=0.05)

    dense3_units = st.slider("Dense Layer 3 Units", 32, 256, 128, step=16)
    dropout3_rate = st.slider("Dropout Rate after Dense 3", 0.0, 0.5, 0.3, step=0.05)

    epochs = st.slider("Epochs", 1, 50, 10)
    batch_size = st.slider("Batch Size", 8, 32, 128, step=8)
    if st.button("Train Model with LSTM"):
        lstm(lstm_units=lstm_units, 
             dense1_units=dense1_units, 
             dropout1_rate=dropout1_rate,
             dense2_units=dense2_units, 
             dropout2_rate=dropout2_rate, 
             dense3_units=dense3_units, 
             dropout3_rate=dropout3_rate, 
             epochs=epochs, 
             batch_size=batch_size,
             bidirectional=bidirectional)
else:
    st.info("Please preprocess the data first before training.")
