import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# ==============================
# LOAD MODEL & SCALER
# ==============================
@st.cache_resource
def load_lstm_model():
    return load_model("model_bbca.h5")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_lstm_model()
scaler = load_scaler()

# ==============================
# PREDIKSI 1 LANGKAH KE DEPAN
# ==============================
def predict_next(last_window):
    """
    last_window: list/array ukuran (window,) berisi closing price terakhir
    """
    scaled = scaler.transform(np.array(last_window).reshape(-1, 1))
    X = scaled.reshape(1, len(last_window), 1)
    pred_scaled = model.predict(X)
    pred = scaler.inverse_transform(pred_scaled)[0][0]
    return float(pred)

# ==============================
# STREAMLIT APP UI
# ==============================
st.set_page_config(page_title="BBCA LSTM Forecast", layout="wide")

st.title("Prediksi Harga Saham BBCA Menggunakan LSTM")
st.markdown("Aplikasi ini menampilkan prediksi harga berikutnya berdasarkan model LSTM yang telah dilatih.")

uploaded = st.file_uploader("Upload data harga BBCA (CSV, kolom: Date & Close):", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    # Pastikan kolom yang benar
    if "Close" not in df.columns:
        st.error("CSV harus memiliki kolom 'Close'")
    else:
        # Pastikan sorting
        df = df.sort_values("Date")

        st.subheader("Grafik Harga BBCA")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close Price"))
        st.plotly_chart(fig, use_container_width=True)

        # Input window
        st.subheader(" Pengaturan Prediksi")
        window = st.slider("Window Size (harus sama seperti saat training!)", 10, 60, 30)

        if len(df) < window:
            st.error(f"Data minimal harus {window} hari terakhir.")
        else:
            # ambil window terakhir
            last_window = df["Close"].values[-window:]
            next_price = predict_next(last_window)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Harga Terakhir", f"{last_window[-1]:,.2f} IDR")

            with col2:
                st.metric("Prediksi Harga Berikutnya", f"{next_price:,.2f} IDR")

            st.success("Prediksi berhasil dihitung!")

else:
    st.info("Silakan upload file CSV untuk memulai prediksi.")
