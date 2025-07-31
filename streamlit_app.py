import streamlit as st

predict_page = st.Page("predict.py", title="Prediction", icon="🎉")

pg = st.navigation([predict_page])


pg.run()

