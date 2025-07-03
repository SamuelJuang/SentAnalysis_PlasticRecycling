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

    model = load_model('lstm_model.h5')
    X_test_pad = st.session_state.X_test_pad
    y_test = st.session_state.y_test
    bar.progress(0.1)

   # Predict probabilities
    y_pred = model.predict(X_test_pad)

    # Convert to binary predictions with threshold 0.5
    y_pred_binary = (y_pred > 0.5).astype(int)

    total_binary_accuracy = (y_pred_binary == y_test).mean()
    print(f"Overall Binary Accuracy (per label, averaged): {total_binary_accuracy:.4f}")

#     # Show classification reports
#     st.subheader("Classification Reports")
#     label_categories = {
#     "Recyclability": ["Negative", "Neutral", "Positive"],
#     "Recyclability (PET)": ["Negative", "Neutral", "Positive"],
#     "Recycling": ["Negative", "Neutral", "Positive"],
#     "Future": ["Negative", "Neutral", "Positive"]
# }
#     st.text("Recyclability:")
#     st.code(classification_report(
#         y_true_recycle,
#         y_pred_recycle,
#         target_names=label_categories['Recyclability']
#     ))

#     st.text("Recyclability (PET):")
#     st.code(classification_report(
#         y_true_pet,
#         y_pred_pet,
#         target_names=label_categories['Recyclability (PET)']
#     ))

#     st.text("Recycling:")
#     st.code(classification_report(
#         y_true_process,
#         y_pred_process,
#         target_names=label_categories['Recycling']
#     ))
#     bar.progress(0.5)
#     st.text("Future:")
#     st.code(classification_report(
#         y_true_future,
#         y_pred_future,
#         target_names=label_categories['Future']
#     ))

    st.subheader("Confusion Matrices")
     # Recyclability
    # Convert probabilities to binary predictions with threshold
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Get confusion matrix for each label
    cm = multilabel_confusion_matrix(y_test, y_pred_binary)

    # Optional: Label names (update to your specific label structure)
    label_names = [
        "Recycling_Negative", "Recycling_Neutral", "Recycling_Positive",
        "Recyclability (PET)_Negative", "Recyclability (PET)_Neutral", "Recyclability (PET)_Positive",
        "Recyclability_Negative", "Recyclability_Neutral", "Recyclability_Positive",
        "Future_Negative", "Future_Neutral", "Future_Positive"
    ]

    # Plot confusion matrix for each label
    for idx, matrix in enumerate(cm):
        st.subheader(f"Confusion Matrix for {label_names[idx]}")
        fig_conf = plt.figure(figsize=(4, 3))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"Confusion Matrix for {label_names[idx]}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig_conf)
        
        
        


if(os.path.exists('lstm_model.h5') and st.session_state.get('X_test_pad') is not None):
    st.title("Model Evaluation")
    evaluate_model()
else:
    st.title("Model Evaluation")
    st.write("No model found. Please train the model first.")
    st.stop()