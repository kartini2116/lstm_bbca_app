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
# MENU 2: PREDIKSI BBCA DENGAN MODEL LSTM KAMU
# =====================================================================
if menu == "Prediksi Harga BBCA ":

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
