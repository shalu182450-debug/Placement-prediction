import streamlit as st
import joblib
import numpy as np
import pandas as pd

#going to load model
model=joblib.load("model_lr.joblib")

st.title("Model Deployment using Streamlit")
# adding cgpa input box
cgpa=st.number_input("ENTER YOUR CGPA:  ")

# Adding Button for Prediction
if st.button("PREDICT"):
    cgpa=np.array([[cgpa]])
    prediction=model.predict(cgpa)
    st.success(f"YOUR PREDICTED PACKAGE WILL BE: {prediction}")
