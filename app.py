import streamlit as st
import joblib
import pandas as pd

# Load model
model_rf = joblib.load('random_forest_model.pkl')
model_xgb = joblib.load('xgboost_model.pkl')

st.title("Aplikasi Prediksi dengan Random Forest dan XGBoost")

# Input data
input_data = st.text_input("Masukkan data fitur (misal: 5, 3, 1.5, 0.2)")

if st.button("Prediksi Random Forest"):
    features = [float(x) for x in input_data.split(',')]
    prediction_rf = model_rf.predict([features])
    st.write(f"Prediksi Random Forest: {prediction_rf[0]}")

if st.button("Prediksi XGBoost"):
    features = [float(x) for x in input_data.split(',')]
    prediction_xgb = model_xgb.predict([features])
    st.write(f"Prediksi XGBoost: {prediction_xgb[0]}")