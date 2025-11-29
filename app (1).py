import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# =========================================
# LOAD MODEL & SCALER
# =========================================
MODEL_PATH = "model_bbca.keras"
SCALER_PATH = "scaler_bbca.pkl"

model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# =========================================
# LOAD DATA BBCA (HANYA CLOSE)
# =========================================
df = pd.read_csv("bbca.csv")
close_prices = df["Close"].astype(float).values.reshape(-1, 1)

scaled_data = scaler.transform(close_prices)
TIMESTEP = 60  # harus sama seperti training


# =========================================
# STREAMLIT UI
# =========================================
st.title("📈 Prediksi Harga Saham BBCA Menggunakan LSTM")
st.write("Aplikasi ini memprediksi harga beberapa hari ke depan menggunakan model LSTM yang sudah Anda latih.")

n_days = st.slider("Prediksi berapa hari ke depan?", 1, 30, 7)


# =========================================
# MULTI-STEP FORECASTING
# =========================================
last_sequence = scaled_data[-TIMESTEP:]
current_input = last_sequence.reshape(1, TIMESTEP, 1)

pred_scaled_list = []
pred_real_list = []

for _ in range(n_days):

    next_scaled = model.predict(current_input, verbose=0)[0][0]
    pred_scaled_list.append(next_scaled)

    # ubah kembali ke harga asli
    next_real = scaler.inverse_transform([[next_scaled]])[0][0]
    pred_real_list.append(next_real)

    # update input sequence
    new_seq = np.append(current_input.flatten()[1:], next_scaled)
    current_input = new_seq.reshape(1, TIMESTEP, 1)


# =========================================
# OUTPUT ANGKA PREDIKSI
# =========================================
st.subheader("📌 Hasil Prediksi Harga BBCA")
for i, p in enumerate(pred_real_list, start=1):
    st.write(f"Hari ke-{i}: **Rp {p:,.2f}**")


# =========================================
# PLOTTING MATPLOTLIB
# =========================================
st.subheader("📊 Grafik Harga Aktual & Prediksi")

fig, ax = plt.subplots(figsize=(10, 5))

# plot harga aktual
ax.plot(df["Close"].values, label="Harga Aktual", linewidth=2)

# plot prediksi future
future_index = np.arange(len(df), len(df) + n_days)

ax.plot(future_index, pred_real_list, 
        label="Prediksi", marker="o", linestyle="--", linewidth=2)

ax.set_title("Prediksi Harga Saham BBCA Menggunakan LSTM")
ax.set_xlabel("Index Waktu")
ax.set_ylabel("Harga (IDR)")
ax.legend()
ax.grid(True)

st.pyplot(fig)
