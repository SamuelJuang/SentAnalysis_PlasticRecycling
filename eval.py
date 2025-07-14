import os
import streamlit as st
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns

def evaluate_model():
    status = st.empty()
    bar = st.progress(0)
    status.text("Loading model...")

    model = load_model('bilstm_model.keras')
    X_test_pad = st.session_state.X_test_pad
    y_test = st.session_state.y_test
    bar.progress(1.0)
    status.text("Model loaded successfully.")

    y_pred = model.predict(X_test_pad)
    y_pred_binary = (y_pred > 0.5).astype(int)

    total_binary_accuracy = (y_pred_binary == y_test).mean()
    st.text(f"Overall Accuracy Averaged per Label: {total_binary_accuracy:.4f}")

    st.subheader("Confusion Matrices")

    cm = multilabel_confusion_matrix(y_test, y_pred_binary)

    label_names = [
        "Recyclability_Negative", "Recyclability_Neutral", "Recyclability_Positive",
        "Recyclability (PET)_Negative", "Recyclability (PET)_Neutral", "Recyclability (PET)_Positive",
        "Recycling_Negative", "Recycling_Neutral", "Recycling_Positive",
        "Future_Negative", "Future_Neutral", "Future_Positive"
    ]

    for idx, matrix in enumerate(cm):
        st.subheader(f"Confusion Matrix for {label_names[idx]}")
        fig_conf = plt.figure(figsize=(4, 3))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"Confusion Matrix for {label_names[idx]}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig_conf)
        
        
        


if(os.path.exists('bilstm_model.h5') and st.session_state.get('X_test_pad') is not None):
    st.title("Model Evaluation")
    evaluate_model()
else:
    st.title("Model Evaluation")
    st.write("No model found. Please train the model first.")
    st.stop()