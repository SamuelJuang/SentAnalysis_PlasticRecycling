import streamlit as st

# Define the pages
data_page = st.Page("streamlit.py", title="Data Page", icon="🎈")
machine_page = st.Page("machine.py", title="Model Training", icon="❄️")
eval_page = st.Page("eval.py", title="Evaluation", icon="📊")
predict_page = st.Page("predict.py", title="Prediction", icon="🎉")


# Set up navigation
pg = st.navigation([data_page,machine_page ,eval_page, predict_page])

# Run the selected page
pg.run()