import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ===========================================================
# LOAD MODEL DAN SCALER
# ===========================================================
MODEL_PATH = "model_bbca.keras"
SCALER_PATH = "scaler_bbca.pkl"

model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler: MinMaxScaler = pickle.load(f)

TIMESTEP = 60  # HARUS SAMA DENGAN TRAINING


# ===========================================================
# FUNGSI UNTUK MEMBUAT SEKUENS DATA
# ===========================================================
def create_sequences(data, timestep):
    X, y = [], []
    for i in range(timestep, len(data)):
        X.append(data[i - timestep:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# ===========================================================
# FUNGSI PREDIKSI LSTM
# ===========================================================
def predict_lstm(dataframe, n_future):
    close_prices = dataframe[['Close']].astype(float).values
    scaled = scaler.transform(close_prices)

    last_seq = scaled[-TIMESTEP:]
    seq = last_seq.reshape(1, TIMESTEP, 1)

    preds_scaled = []
    preds_real = []

    for _ in range(n_future):
        next_scaled = model.predict(seq, verbose=0)[0][0]
        preds_scaled.append(next_scaled)

        next_real = scaler.inverse_transform([[next_scaled]])[0][0]
        preds_real.append(next_real)

        new_seq = np.append(seq.flatten()[1:], next_scaled)
        seq = new_seq.reshape(1, TIMESTEP, 1)

    return preds_real


# ===========================================================
# STREAMLIT UI
# ===========================================================
st.title("Prediksi Harga Saham BBCA Menggunakan LSTM")
st.write("Aplikasi ini memprediksi harga saham BBCA untuk beberapa hari ke depan.")

st.sidebar.header("⚙ Pengaturan Prediksi")
n_days = st.sidebar.slider("Prediksi berapa hari ke depan?", 1, 30, 7)

uploaded_file = st.file_uploader("Upload file CSV harga saham (opsional)", type=["csv"])

# Jika user upload dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File berhasil diupload!")
else:
    df = pd.read_csv(DATA_PATH)
    st.info("Menggunakan data default: bbca.csv")

# Validasi kolom Close
if "Close" not in df.columns:
    st.error("CSV harus memiliki kolom 'Close'.")
    st.stop()

st.subheader("Data Harga BBCA (5 Baris Terakhir)")
st.dataframe(df.tail())

# Tombol prediksi
if st.button("Prediksi"):
    preds = predict_lstm(df, n_days)

    st.subheader("Hasil Prediksi LSTM")
    for i, p in enumerate(preds, start=1):
        st.write(f"Hari ke-{i}: **Rp {p:,.2f}**")

    # ======================================
    # GRAFIK MATPLOTLIB
    # ======================================
    st.subheader("Grafik Aktual vs Prediksi")

    fig, ax = plt.subplots(figsize=(10,5))

    actual = df["Close"].astype(float).values
    future_index = np.arange(len(actual), len(actual) + n_days)

    ax.plot(actual, label="Harga Aktual", linewidth=2)
    ax.plot(future_index, preds, label="Prediksi", linestyle="--", marker="o")

    ax.set_title("Prediksi Harga Saham BBCA (LSTM)")
    ax.set_xlabel("Index Waktu")
    ax.set_ylabel("Harga (IDR)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # Tabel prediksi
    pred_df = pd.DataFrame({
        "Hari Ke": np.arange(1, n_days+1),
        "Prediksi Harga (IDR)": preds
    })

    st.subheader("Tabel Hasil Prediksi")
    st.dataframe(pred_df)
