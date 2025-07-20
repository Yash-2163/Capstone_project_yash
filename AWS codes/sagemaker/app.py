# app.py
import streamlit as st
import pandas as pd
import joblib
import preprocessing


# Load the model pipeline once
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🚀 Lead Conversion Predictor")
uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head())

    if st.button("Predict"):
        preds = model.predict(df)
        df["Converted?"] = ["✅" if p else "❌" for p in preds]
        st.subheader("Results")
        st.dataframe(df)

