import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

MODEL_PATH = "model_bbca.keras"
SCALER_PATH = "scaler_bbca.pkl"
DATA_PATH   = "bbca.csv"

model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler: MinMaxScaler = pickle.load(f)

TIMESTEP = 60

st.title("🔮 Prediksi Harga Saham BBCA (LSTM)")

st.sidebar.header("Pengaturan Prediksi")
n_days = st.sidebar.slider("Prediksi berapa hari ke depan?", 1, 30, 7)

# Upload file
uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"])

df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(DATA_PATH)

if "Close" not in df.columns:
    st.error("CSV harus memiliki kolom 'Close'.")
    st.stop()

close_prices = df[['Close']].astype(float).values
scaled = scaler.transform(close_prices)

def predict_lstm(n_future):
    last_seq = scaled[-TIMESTEP:]
    seq = last_seq.reshape(1, TIMESTEP, 1)

    preds_real = []

    for _ in range(n_future):
        next_scaled = model.predict(seq, verbose=0)[0][0]
        next_real = scaler.inverse_transform([[next_scaled]])[0][0]
        preds_real.append(next_real)

        new_seq = np.append(seq.flatten()[1:], next_scaled)
        seq = new_seq.reshape(1, TIMESTEP, 1)

    return preds_real

if st.button("🔍 Jalankan Prediksi"):
    preds = predict_lstm(n_days)
    st.success("Prediksi berhasil!")

    st.subheader("Hasil Prediksi")
    for i, p in enumerate(preds, start=1):
        st.write(f"Hari ke-{i}: **Rp {p:,.2f}**")

    # === Plotly Chart ===
    st.subheader("Grafik Interaktif")
    actual = df["Close"].values
    future_index = np.arange(len(actual), len(actual) + n_days)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual, mode='lines', name='Aktual'))
    fig.add_trace(go.Scatter(x=future_index, y=preds, mode='lines+markers', name='Prediksi'))

    fig.update_layout(
        title="Prediksi Harga BBCA",
        xaxis_title="Index Waktu",
        yaxis_title="Harga (IDR)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Tabel prediksi
    pred_df = pd.DataFrame({
        "Hari Ke": np.arange(1, n_days+1),
        "Prediksi Harga": preds
    })
    st.dataframe(pred_df)
