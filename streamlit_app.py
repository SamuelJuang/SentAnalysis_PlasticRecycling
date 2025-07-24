import streamlit as st

data_page = st.Page("streamlit.py", title="Data Page", icon="🎈")
machine_page = st.Page("machine.py", title="Model Training", icon="❄️")
eval_page = st.Page("eval.py", title="Evaluation", icon="📊")
predict_page = st.Page("predict.py", title="Prediction", icon="🎉")



pg = st.navigation([predict_page])


pg.run()

