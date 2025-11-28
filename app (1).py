import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from tensorflow.keras.models import load_model
import pickle

# ============================
# LOAD SCALER + MODEL
# ============================

scaler = pickle.load(open("scaler_bbca.pkl", "rb"))
model = load_model("model_lstm_bbca.h5")

# ============================
# FUNGSI PREDIKSI LSTM
# ============================

def predict_next(data, window=60):
    """
    data = array close price
    window = sequence length
    """
    scaled = scaler.transform(data.reshape(-1, 1))
    
    X = scaled[-window:].reshape(1, window, 1)
    pred_scaled = model.predict(X)
    pred = scaler.inverse_transform(pred_scaled)
    
    return float(pred)


# ============================
# STREAMLIT UI
# ============================

st.set_page_config(
    page_title="Real-Time Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Real-Time LSTM Stock Prediction Dashboard")
st.write("Dashboard interaktif untuk memprediksi harga saham menggunakan model LSTM Anda.")

# ============================
# SIDEBAR PARAMETER
# ============================

st.sidebar.header("⚙️ Chart Parameters")

# Input ticker
ticker = st.sidebar.text_input("Ticker", value="BBCA.JK")

# Time period
period = st.sidebar.selectbox(
    "Time Period",
    ["1d", "5d", "1mo", "6mo", "1y", "5y", "max"],
    index=3
)

# Chart type
chart_type = st.sidebar.selectbox(
    "Chart Type",
    ["Line", "Candlestick"]
)

# Window size (sequence length)
window = st.sidebar.slider("Window Size (LSTM)", 30, 120, 60)

# Predict button
run = st.sidebar.button("Update Chart")

# ============================
# DATA LOADING
# ============================

st.subheader(f"📊 Data Harga Saham: {ticker}")

data = yf.download(ticker, period=period)

if len(data) == 0:
    st.error("Ticker tidak valid atau data tidak ditemukan.")
    st.stop()


# ============================
# GRAPH AREA
# ============================

tab1, tab2 = st.tabs(["📈 Chart", "🔮 Prediksi"])

with tab1:
    st.markdown("### Grafik Harga")

    fig = go.Figure()

    if chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Close Price"
        ))

    elif chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Candlestick"
        ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Date",
        yaxis_title="Price",
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 🔮 Prediksi Harga Berikutnya")

    last_close = data["Close"].values
    next_price = predict_next(last_close, window)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="📌 Harga Sekarang",
            value=f"{last_close[-1]:,.2f} IDR"
        )
    with col2:
        st.metric(
            label="🔮 Prediksi Berikutnya",
            value=f"{next_price:,.2f} IDR",
            delta=next_price - last_close[-1]
        )

    st.write("Prediksi didasarkan pada 60 data terakhir (configurable).")


