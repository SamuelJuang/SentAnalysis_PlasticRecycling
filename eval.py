import os
import streamlit as st
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate_model():
    status = st.empty()
    bar = st.progress(0)
    status.text("Loading model...")

    model = load_model('lstm_model.h5')
    X_test_pad = st.session_state.X_test_pad
    y_recycle_test = st.session_state.y_recycle_test
    y_pet_test = st.session_state.y_pet_test
    y_process_test = st.session_state.y_process_test
    y_effective_test = st.session_state.y_effective_test
    bar.progress(0.1)

    eval_result = model.evaluate(
    X_test_pad,
    {
        'recycle': y_recycle_test,
        'pet': y_pet_test,
        'process': y_process_test,
        'future': y_effective_test
    }
    )
    # Predict
    y_pred = model.predict(X_test_pad)
    y_pred_recycle = np.argmax(y_pred[0], axis=1)
    y_pred_pet = np.argmax(y_pred[1], axis=1)
    y_pred_process = np.argmax(y_pred[2], axis=1)
    y_pred_future = np.argmax(y_pred[3], axis=1)

    # Actual values
    y_true_recycle = np.argmax(y_recycle_test.values, axis=1)
    y_true_pet = np.argmax(y_pet_test.values, axis=1)
    y_true_process = np.argmax(y_process_test.values, axis=1)
    y_true_future = np.argmax(y_effective_test.values, axis=1)

    # Show classification reports
    st.subheader("Classification Reports")
    label_categories = {
    "Recyclability": ["Negative", "Neutral", "Positive"],
    "Recyclability (PET)": ["Negative", "Neutral", "Positive"],
    "Recycling": ["Negative", "Neutral", "Positive"],
    "Future": ["Negative", "Neutral", "Positive"]
}
    st.text("Recyclability:")
    st.code(classification_report(
        y_true_recycle,
        y_pred_recycle,
        target_names=label_categories['Recyclability']
    ))

    st.text("Recyclability (PET):")
    st.code(classification_report(
        y_true_pet,
        y_pred_pet,
        target_names=label_categories['Recyclability (PET)']
    ))

    st.text("Recycling:")
    st.code(classification_report(
        y_true_process,
        y_pred_process,
        target_names=label_categories['Recycling']
    ))
    bar.progress(0.5)
    st.text("Future:")
    st.code(classification_report(
        y_true_future,
        y_pred_future,
        target_names=label_categories['Future']
    ))

    st.subheader("Confusion Matrices")
     # Recyclability
    cm_recycle = confusion_matrix(np.argmax(y_recycle_test, axis=1), y_pred_recycle)
    disp_recycle = ConfusionMatrixDisplay(confusion_matrix=cm_recycle, display_labels=label_categories['Recyclability'])
    fig1, ax1 = plt.subplots()
    disp_recycle.plot(cmap='Blues', ax=ax1)
    plt.title("Recyclability Confusion Matrix")
    st.pyplot(fig1)

    # Recyclability (PET)
    cm_pet = confusion_matrix(np.argmax(y_pet_test, axis=1), y_pred_pet)
    disp_pet = ConfusionMatrixDisplay(confusion_matrix=cm_pet, display_labels=label_categories['Recyclability (PET)'])
    fig2, ax2 = plt.subplots()
    disp_pet.plot(cmap='Blues', ax=ax2)
    plt.title("Recyclability (PET) Confusion Matrix")
    st.pyplot(fig2)

    # Recycling
    cm_process = confusion_matrix(np.argmax(y_process_test, axis=1), y_pred_process)
    disp_process = ConfusionMatrixDisplay(confusion_matrix=cm_process, display_labels=label_categories['Recycling'])
    fig3, ax3 = plt.subplots()
    disp_process.plot(cmap='Blues', ax=ax3)
    plt.title("Recycling Confusion Matrix")
    st.pyplot(fig3)

    # Future
    cm_future = confusion_matrix(np.argmax(y_effective_test, axis=1), y_pred_future)
    disp_future = ConfusionMatrixDisplay(confusion_matrix=cm_future, display_labels=label_categories['Future'])
    fig4, ax4 = plt.subplots()
    disp_future.plot(cmap='Blues', ax=ax4)
    plt.title("Future Confusion Matrix")
    st.pyplot(fig4)
    bar.progress(1.0)
    status.success("Model evaluation complete!")
    


if(os.path.exists('lstm_model.h5') and st.session_state.get('X_test_pad') is not None):
    st.title("Model Evaluation")
    evaluate_model()
else:
    st.title("Model Evaluation")
    st.write("No model found. Please train the model first.")
    st.stop()