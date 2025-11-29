# =====================================================================
#   DASHBOARD + MODEL LSTM BBCA (KODE SUDAH DIMODIFIKASI UNTUK ANDA)
# =====================================================================

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import ta
import pickle
from tensorflow.keras.models import load_model

# =====================================================================
# LOAD MODEL & SCALER UNTUK PREDIKSI BBCA
# =====================================================================
MODEL_PATH = "model_bbca.keras"
SCALER_PATH = "scaler_bbca.pkl"
DATA_PATH   = "bbca.csv"

model = load_model(MODEL_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

TIMESTEP = 60   # harus sama seperti model training

# =====================================================================
# FUNCTION PREDIKSI LSTM BBCA
# =====================================================================
def predict_lstm(n_days):
    df = pd.read_csv(DATA_PATH)
    close_prices = df["Close"].astype(float).values.reshape(-1, 1)

    scaled = scaler.transform(close_prices)

    seq = scaled[-TIMESTEP:]
    seq = seq.reshape(1, TIMESTEP, 1)

    pred_scaled_list = []
    pred_real_list = []

    for _ in range(n_days):
        next_scaled = model.predict(seq, verbose=0)[0][0]
        pred_scaled_list.append(next_scaled)

        next_real = scaler.inverse_transform([[next_scaled]])[0][0]
        pred_real_list.append(next_real)

        new_seq = np.append(seq.flatten()[1:], next_scaled)
        seq = new_seq.reshape(1, TIMESTEP, 1)

    return df, pred_real_list


# =====================================================================
#  FUNCTION REAL-TIME STOCK FROM YFINANCE
# =====================================================================
def fetch_stock_data(ticker, period, interval):
    end_date = datetime.now()
    if period == '1wk':
        start_date = end_date - timedelta(days=7)
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        data = yf.download(ticker, period=period, interval=interval)
    return data

def process_data(data):
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data

def calculate_metrics(data):
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = data['High'].max()
    low = data['Low'].min()
    volume = data['Volume'].sum()
    return last_close, change, pct_change, high, low, volume

def add_technical_indicators(data):
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    return data


# =====================================================================
# STREAMLIT LAYOUT
# =====================================================================
st.set_page_config(layout="wide")
st.title('Real Time Stock Dashboard + LSTM BBCA Forecast')

# Sidebar
st.sidebar.header('Navigasi')
menu = st.sidebar.radio("Pilih Menu", [
    "📊 Real-Time Stock Chart",
    "🔮 Prediksi Harga BBCA (LSTM Model Anda)"
])

# Mapping
interval_mapping = {
    '1d': '1m',
    '1wk': '30m',
    '1mo': '1d',
    '1y': '1wk',
    'max': '1wk'
}



# =====================================================================
# MENU 1: REAL-TIME CHART
# =====================================================================
if menu == "📊 Real-Time Stock Chart":

    st.sidebar.header("Chart Parameters")
    ticker = st.sidebar.text_input('Ticker', 'ADBE')
    time_period = st.sidebar.selectbox('Time Period', ['1d', '1wk', '1mo', '1y', 'max'])
    chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])
    indicators = st.sidebar.multiselect('Technical Indicators', ['SMA 20', 'EMA 20'])

    if st.sidebar.button("Update Chart"):

        data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
        data = process_data(data)
        data = add_technical_indicators(data)
        
        last_close, change, pct_change, high, low, volume = calculate_metrics(data)

        st.metric(f"{ticker} Last Price", f"{last_close:.2f}", f"{change:.2f} ({pct_change:.2f}%)")

        # Plot
        fig = go.Figure()
        if chart_type == 'Candlestick':
            fig.add_trace(go.Candlestick(x=data['Datetime'], open=data['Open'],
                                         high=data['High'], low=data['Low'], close=data['Close']))
        else:
            fig = px.line(data, x='Datetime', y='Close')

        for indicator in indicators:
            if indicator == 'SMA 20':
                fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'], name='SMA 20'))
            elif indicator == 'EMA 20':
                fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'], name='EMA 20'))

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)



# =====================================================================
# MENU 2: PREDIKSI BBCA DENGAN MODEL LSTM KAMU
# =====================================================================
if menu == "🔮 Prediksi Harga BBCA (LSTM Model Anda)":

    st.header("Prediksi Harga Saham BBCA Menggunakan Model LSTM Anda")

    n_days = st.slider("Prediksi berapa hari ke depan?", 1, 30, 7)

    if st.button("Prediksi"):

        df, pred_list = predict_lstm(n_days)

        st.subheader("Hasil Prediksi")
        for i, p in enumerate(pred_list, start=1):
            st.write(f"Hari ke-{i}: **Rp {p:,.2f}**")

        # Plot dengan Plotly
        future_idx = list(range(len(df), len(df)+n_days))

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df["Close"],
            mode="lines",
            name="Harga Aktual"
        ))

        fig_pred.add_trace(go.Scatter(
            x=future_idx,
            y=pred_list,
            mode="lines+markers",
            name="Prediksi LSTM"
        ))

        fig_pred.update_layout(
            title="Prediksi Harga BBCA (LSTM)",
            xaxis_title="Index Waktu",
            yaxis_title="Harga (IDR)",
            height=600
        )

        st.plotly_chart(fig_pred, use_container_width=True)
