import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ---------------------------------------
#  LOAD MODEL & SCALER
# ---------------------------------------
MODEL_PATH = "model_bbca.h5"
SCALER_PATH = "scaler_bbca.pkl"

model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ---------------------------------------
#  LOAD DATA BBCA
# ---------------------------------------
df = pd.read_csv("bbca.csv")
close_prices = df["Close"].astype(float).values.reshape(-1, 1)

scaled_data = scaler.transform(close_prices)
TIMESTEP = 60  # harus sesuai training


# ---------------------------------------
#  STREAMLIT UI
# ---------------------------------------
st.title("Prediksi Harga Saham BBCA Menggunakan LSTM (Matplotlib)")
st.write("Aplikasi ini menggunakan model LSTM yang sudah Anda latih.")

n_days = st.slider("Prediksi berapa hari ke depan?", 1, 30, 7)


# ---------------------------------------
#  MULTI-STEP FORECASTING
# ---------------------------------------
last_sequence = scaled_data[-TIMESTEP:]
current_input = last_sequence.reshape(1, TIMESTEP, 1)

pred_scaled_list = []
pred_real_list = []

for _ in range(n_days):
    next_scaled = model.predict(current_input)[0][0]
    pred_scaled_list.append(next_scaled)

    # ubah ke harga asli
    next_real = scaler.inverse_transform([[next_scaled]])[0][0]
    pred_real_list.append(next_real)

    # update sequence input
    new_seq = np.append(current_input.flatten()[1:], next_scaled)
    current_input = new_seq.reshape(1, TIMESTEP, 1)


# ---------------------------------------
#  OUTPUT ANGKA PREDIKSI
# ---------------------------------------
st.subheader("Hasil Prediksi")
for i, p in enumerate(pred_real_list, start=1):
    st.write(f"Hari ke-{i}: **Rp {p:,.2f}**")


# ---------------------------------------
#  GRAFIK MATPLOTLIB
# ---------------------------------------
st.subheader("Grafik Harga & Prediksi LSTM")

fig, ax = plt.subplots(figsize=(10, 5))

# plot harga aktual
ax.plot(df["Close"].values, label="Harga Aktual", linewidth=2)

# future index untuk plot prediksi
future_index = np.arange(len(df), len(df) + n_days)

# plot prediksi
ax.plot(future_index, pred_real_list, label="Prediksi", marker="o", linestyle="--")

ax.set_title("Prediksi Harga BBCA Menggunakan LSTM")
ax.set_xlabel("Index Waktu")
ax.set_ylabel("Harga (IDR)")
ax.legend()
ax.grid(True)

st.pyplot(fig).models 

import load_model
import pickle
import matplotlib.pyplot as plt

# =============================
# Load model dan scaler
# =============================
model = load_model("model_bbca.keras")

with open("scaler_bbca.pkl", "rb") as f:
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
