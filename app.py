import streamlit as st
import joblib

# Load model
model = joblib.load("lightgbm_final.pkl")

st.title("Crop Yield Prediction App")

st.write("Model loaded successfully!")