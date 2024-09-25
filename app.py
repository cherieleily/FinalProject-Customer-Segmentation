import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained models
rf_model = joblib.load('random_forest_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')

# Simpan label encoder untuk fitur yang bersifat kategorikal
gender_encoder = LabelEncoder()
gender_encoder.classes_ = np.array(['Female', 'Male'])

married_encoder = LabelEncoder()
married_encoder.classes_ = np.array(['No', 'Yes'])

graduated_encoder = LabelEncoder()
graduated_encoder.classes_ = np.array(['No', 'Yes'])

profession_encoder = LabelEncoder()
profession_encoder.classes_ = np.array(['Artist', 'Doctor', 'Engineer', 'Entertainment', 'Healthcare', 'Lawyer'])

# Input dari pengguna
st.title("Aplikasi Prediksi dengan Random Forest dan XGBoost")

gender = st.selectbox("Jenis Kelamin", ('Male', 'Female'))
ever_married = st.selectbox("Pernah Menikah", ('Yes', 'No'))
age = st.number_input("Umur", min_value=18, max_value=100, value=30)
graduated = st.selectbox("Lulusan Universitas", ('Yes', 'No'))
profession = st.selectbox("Profesi", ('Artist', 'Doctor', 'Engineer', 'Entertainment', 'Healthcare', 'Lawyer'))
work_experience = st.number_input("Pengalaman Kerja (Tahun)", min_value=0, max_value=40, value=5)
spending_score = st.selectbox("Spending Score", ('Low', 'Average', 'High'))
family_size = st.number_input("Ukuran Keluarga", min_value=1, max_value=10, value=3)
var_1 = st.selectbox("Kategori Var_1", ('Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5', 'Cat_6', 'Cat_7'))

# Transformasi input kategorikal ke bentuk numerik
gender_encoded = gender_encoder.transform([gender])[0]
married_encoded = married_encoder.transform([ever_married])[0]
graduated_encoded = graduated_encoder.transform([graduated])[0]
profession_encoded = profession_encoder.transform([profession])[0]

# Masukkan fitur ke dalam array
features = [gender_encoded, married_encoded, age, graduated_encoded, profession_encoded, work_experience, spending_score, family_size, var_1]

# Prediksi Random Forest
if st.button('Prediksi Random Forest'):
    prediction_rf = rf_model.predict([features])
    st.write(f'Prediksi Random Forest: {prediction_rf[0]}')

# Prediksi XGBoost
if st.button('Prediksi XGBoost'):
    prediction_xgb = xgb_model.predict([features])
    st.write(f'Prediksi XGBoost: {prediction_xgb[0]}')