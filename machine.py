from sklearn.model_selection import train_test_split
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional,BatchNormalization,GlobalMaxPooling1D
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
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

def lstm(lstm_units,re_dropout, dense_units,dropout_rate, epoch_num):
         
    if(os.path.exists('oversampled_data.csv')):
        df_encoded = pd.read_csv('oversampled_data.csv')
    else:
        df_encoded = pd.read_csv('cleaned_and_normalized_data.csv')
    status = st.empty()
    bar = st.progress(0)

    # TRAIN TEST SPLIT
    status.text("Splitting the data into train and test sets...")
    bar.progress(0.1)

    # Combine all target columns into a single multi-label matrix
    target_columns = [
        "Recycling_Negative", "Recycling_Neutral", "Recycling_Positive",
        "Recyclability (PET)_Negative", "Recyclability (PET)_Neutral", "Recyclability (PET)_Positive",
        "Recyclability_Negative", "Recyclability_Neutral", "Recyclability_Positive",
        "Future_Negative", "Future_Neutral", "Future_Positive"
    ]

    # Features: Assuming 'text' column is your input
    X = df_encoded["text"].values  # Or tokenized form if already processed
    y = df_encoded[target_columns].values  # Shape: (num_samples, 12)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    st.session_state.y_test = y_test

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
    max_len = X_train_pad.shape[1]

    bar.progress(0.5)
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100


    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Input layer for padded sequences
    inputs = Input(shape=(max_len,), name='text_input')

    # Embedding layer
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(inputs)

    # Bidirectional LSTM layer
    x = Bidirectional(LSTM(lstm_units, return_sequences=True, recurrent_dropout= re_dropout))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(dense_units, activation='relu')(x)
    x = BatchNormalization()(x)

    x = Dropout(dropout_rate)(x)


    # Output layer for multi-label (12 classes) with sigmoid activation
    output = Dense(12, activation='sigmoid', name='multi_label_output')(x)

    # Build model
    model = Model(inputs=inputs, outputs=output)

    # Compile with binary crossentropy loss for multi-label
    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])


    model.summary()

    
    bar.progress(0.8)
    # ---- TRAIN ----
    callback = StreamlitProgressBarCallback(epoch_num)
    history = model.fit(X_train_pad, y_train, epochs=epoch_num, batch_size=32, validation_split=0.2,callbacks=[early_stop,callback])

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

    # Accuracy plots 
    st.subheader("Accuracy per Epoch")
    fig_acc = plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig_acc)
   
    # Save the model
    model.save('lstm_model.h5')

    

st.title("Multi-Label LSTM Training")

if os.path.exists('cleaned_and_normalized_data.csv') or os.path.exists('oversampled_data.csv'):

    st.title("Multi-Label LSTM Training")

    lstm_units = st.slider("BiLSTM Units", 32, 256, 64, step=16)
    # bidirectional = st.checkbox("Use Bidirectional LSTM", value=True)
    reccurent_dropout = st.slider("Recurrent Dropout", 0.0, 2.0, 0.5, step=0.05)
    dense_units = st.slider("Dense Layer  Units", 64, 1024, 512, step=64)
    dropout_rate = st.slider("Dropout Rate", 0.0, 0.7, 0.4, step=0.05)

    epochs = st.slider("Epochs", 1, 50, 10)
    batch_size = st.slider("Batch Size", 8,128, 32, step=8)
    if st.button("Train Model with LSTM"):
        lstm(lstm_units=lstm_units, 
             re_dropout=reccurent_dropout,
             dense_units=dense_units, 
             dropout_rate=dropout_rate,
             epoch_num=epochs, 
             )
else:
    st.info("Please preprocess the data first before training.")
