import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
import pickle
import matplotlib.pyplot as plt

# =============================
# Load model dan scaler
# =============================
model = load_model("model_bbca.keras")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =============================
# Streamlit UI
# =============================
st.title("Prediksi Harga Saham BBCA Menggunakan LSTM (Keras)")
st.write("Upload file CSV berisi harga penutupan (Close) untuk melakukan prediksi.")

uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Close" not in df.columns:
        st.error("CSV harus memiliki kolom 'Close'.")
        st.stop()

    st.subheader("Data Harga Saham")
    st.dataframe(df.tail())

    # =============================
    # Preprocessing
    # =============================
    data = df["Close"].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    window = 49  # sama seperti training

    if len(scaled_data) < window:
        st.error(f"Data minimal harus {window} baris.")
        st.stop()

    last_window = scaled_data[-window:]
    X_input = last_window.reshape(1, window, 1)

    # =============================
    # Prediksi
    # =============================
    pred_scaled = model.predict(X_input)
    pred = scaler.inverse_transform(pred_scaled)[0][0]

    st.subheader("Hasil Prediksi")
    st.write(f"**Prediksi Harga Berikutnya: Rp {pred:,.2f}**")

    # =============================
    # Plot
    # =============================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Close"].values, label="Harga Aktual")
    ax.scatter(len(df), pred, label="Prediksi", s=80)
    ax.legend()
    st.pyplot(fig)
