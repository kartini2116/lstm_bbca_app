import streamlit as st
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from tensorflow.keras.models import load_model  # KERAS
import plotly.graph_objects as go

# ========================================
# LOAD MODEL DAN SCALER (CACHED)
# ========================================
@st.cache_resource
def load_lstm_model():
    return load_model("model_bbca.keras")   # jika pakai .keras
    # return load_model("model_bbca.h5")    # jika pakai .h5

@st.cache_resource
def load_scaler():
    return joblib.load("scaler_bbca.pkl")

model = load_lstm_model()
scaler = load_scaler()

# ========================================
# FUNGSI PREDIKSI
# ========================================
def predict_next(window_data):
    """
    window_data: np.array shape (window_size,)
    """
    # memastikan array 2 dimensi
    scaled = scaler.transform(window_data.reshape(-1, 1))

    # reshape ke [1, window, 1] sesuai input LSTM
    X = scaled.reshape(1, len(window_data), 1)

    pred_scaled = model.predict(X)
    pred = scaler.inverse_transform(pred_scaled)[0][0]

    return float(pred)


# ========================================
# STREAMLIT UI
# ========================================
st.set_page_config(page_title="Prediksi Saham BBCA LSTM", layout="wide")
st.title("Prediksi Harga Saham BBCA Menggunakan LSTM (Keras)")
st.write("Aplikasi ini mengambil data otomatis dari Yahoo Finance dan memprediksi harga berikutnya menggunakan model LSTM Anda.")

# ========================================
# SIDEBAR INPUT
# ========================================
st.sidebar.header("⚙️ Pengaturan Prediksi")

ticker = st.sidebar.text_input("Ticker Saham", value="BBCA.JK")
period = st.sidebar.selectbox("Periode Data", ["6mo", "1y", "2y", "5y", "max"], index=1)
window = st.sidebar.slider("Window Size (harus sama dengan training)", 30, 100, 60)

run = st.sidebar.button("Ambil Data & Prediksi")

# ========================================
# KETIKA USER MENEKAN TOMBOL
# ========================================
if run:
    # =======================
    # Ambil data harga
    # =======================
    df = yf.download(ticker, period=period)

    if len(df) == 0:
        st.error("Ticker tidak ditemukan.")
        st.stop()

    df = df.dropna()

    # =======================
    # Tampilkan grafik harga
    # =======================
    st.subheader("Grafik Harga Saham")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close Price"
    ))
    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # pastikan data cukup untuk window
    if len(df) < window:
        st.error(f"Data kurang untuk window size {window}.")
        st.stop()

    # =======================
    # Prediksi harga berikutnya
    # =======================

    last_window = df["Close"].values[-window:]
    next_price = predict_next(last_window)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Harga Penutupan Terakhir",
                  f"{last_window[-1]:,.2f} IDR")

    with col2:
        st.metric("Prediksi Harga Berikutnya",
                  f"{next_price:,.2f} IDR",
                  delta=next_price - last_window[-1])

    st.success("Prediksi berhasil dihitung")

else:
    st.info("Klik tombol *Ambil Data & Prediksi* untuk memulai.")
