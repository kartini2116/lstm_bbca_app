import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
import plotly.graph_objects as go

# =========================================================
# 1. LOAD MODEL DAN SCALER
# =========================================================
MODEL_PATH = "model_bbca.keras"
SCALER_PATH = "scaler_bbca.pkl"

model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)


# =========================================================
# 2. FUNGSI AMBIL DATASET DARI GITHUB
# =========================================================
RAW_URL = "https://raw.githubusercontent.com/USERNAME/REPO/BRANCH/bbca.csv"

def load_data_from_github():
    content = requests.get(RAW_URL).text
    df = pd.read_csv(StringIO(content))
    df["Close"] = df["Close"].astype(float)
    return df


# =========================================================
# 3. PREDIKSI MULTI-STEP TANPA DATA LAIN
# =========================================================
TIMESTEP = model.input_shape[1]   # otomatis baca timesteps model


def predict_future(df, n_days):
    close = df["Close"].values.reshape(-1, 1)
    
    scaled = scaler.transform(close)

    seq = scaled[-TIMESTEP:].reshape(1, TIMESTEP, 1)

    predictions = []

    for _ in range(n_days):
        next_scaled = model.predict(seq, verbose=0)[0][0]
        next_real = scaler.inverse_transform([[next_scaled]])[0][0]
        predictions.append(next_real)

        seq = np.append(seq.flatten()[1:], next_scaled).reshape(1, TIMESTEP, 1)

    return predictions


# =========================================================
# 4. STREAMLIT UI
# =========================================================
st.title("Prediksi Harga BBCA Menggunakan LSTM")

if st.button("Muat Data dari GitHub"):
    df = load_data_from_github()
    st.success("Data berhasil dimuat dari GitHub!")
    st.dataframe(df.tail())

    n_days = st.slider("Prediksi berapa hari ke depan?", 1, 30, 7)

    if st.button("Prediksi"):
        pred = predict_future(df, n_days)

        st.subheader("Hasil Prediksi Harga")

        for i, p in enumerate(pred, start=1):
            st.write(f"Hari ke-{i}: **Rp {p:,.2f}**")

        future_idx = list(range(len(df), len(df) + n_days))

        fig = go.Figure()
        
        # Harga aktual
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df["Close"],
            mode="lines",
            name="Aktual"
        ))

        # Prediksi
        fig.add_trace(go.Scatter(
            x=future_idx,
            y=pred,
            mode="lines+markers",
            name="Prediksi"
        ))

        fig.update_layout(
            title="Prediksi Harga Saham BBCA (LSTM)",
            xaxis_title="Index",
            yaxis_title="Harga",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)
