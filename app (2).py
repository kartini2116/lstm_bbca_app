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
# MENU 2: PREDIKSI BBCA DENGAN MODEL LSTM KAMU
# =====================================================================
if menu == "Prediksi Harga BBCA ":

    st.header("Prediksi Harga Saham BBCA ")

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
