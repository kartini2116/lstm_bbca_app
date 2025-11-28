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
model = load_model("model_bbca.keras")

# ============================
# FUNGSI PREDIKSI LSTM
# ============================

def predict_next(data, window=60):
    """
    data = array close price
    window = sequence length
    """

    # Jika data kurang dari window → ulangi nilai terakhir
    if len(data) < window:
        seq = np.array([data[-1]] * window).reshape(-1, 1)
    else:
        seq = data[-window:].reshape(-1, 1)

    # Scaling
    scaled = scaler.transform(seq)

    # Reshape LSTM
    X = scaled.reshape(1, window, 1)

    # Prediksi
    pred_scaled = model.predict(X)
    pred = scaler.inverse_transform(pred_scaled)

    return float(pred)


# ============================
# STREAMLIT UI SETUP
# ============================

st.set_page_config(
    page_title="Real-Time LSTM Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Real-Time LSTM Stock Prediction Dashboard")
st.write("Dashboard interaktif untuk memprediksi harga saham menggunakan model LSTM Anda.")

# ============================
# SIDEBAR
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

# Window untuk model
window = st.sidebar.slider("Window Size (LSTM)", 30, 120, 60)

# Button
run = st.sidebar.button("Update Chart")

# ============================
# LOAD DATA
# ============================

st.subheader(f"📊 Data Harga Saham: {ticker}")

data = yf.download(ticker, period=period)

if len(data) == 0:
    st.error("Ticker tidak valid atau data tidak ditemukan.")
    st.stop()

# ============================
# TAB GRAPH & PREDICT
# ============================

tab1, tab2 = st.tabs(["📈 Chart", "🔮 Prediksi"])

# ---------------- Chart Tab ----------------
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

# ---------------- Prediction Tab ----------------
with tab2:
    st.markdown("### 🔮 Prediksi Harga Berikutnya Menggunakan LSTM")

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

    st.write(
        f"Prediksi didasarkan pada {window} data terakhir. "
        "Model menggunakan arsitektur LSTM yang dilatih pada data BBCA."
    )
