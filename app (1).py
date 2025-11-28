import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.title("Prediksi Saham BBCA (LSTM)")

# Load model & scaler
model = load_model("model_bbca.keras")

with open("scaler_bbca.pkl", "rb") as f:
    scaler = pickle.load(f)

# Input pengguna
input_val = st.number_input("Masukkan harga penutupan terakhir:", min_value=0.0, value=10000.0)

if st.button("Prediksi"):
    data = np.array([[input_val]])
    data_scaled = scaler.transform(data)
    data_scaled = data_scaled.reshape((1, 1, 1))  # contoh shape
    pred = model.predict(data_scaled)[0][0]
    pred_rescaled = scaler.inverse_transform([[pred]])[0][0]

    st.write(f"📈 **Hasil prediksi harga berikutnya: {pred_rescaled:.2f}**")
