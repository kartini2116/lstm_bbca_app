import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# CONFIG PAGE STYLE
# ==========================================
st.set_page_config(
    page_title="Prediksi Harga BBCA - LSTM",
    page_icon="📈",
    layout="wide"
)

# CSS styling untuk tampilan profesional
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #1F4E79;
}
.sub-title {
    font-size: 20px;
    color: #555;
}
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL & SCALER
# ==========================================
MODEL_PATH = "model_bbca.keras"
SCALER_PATH = "scaler_bbca.pkl"
DATA_PATH   = "bbca.csv"

model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler: MinMaxScaler = pickle.load(f)

TIMESTEP = 60


# ==========================================
# PREDIKSI LSTM
# ==========================================
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


# ==========================================
# STREAMLIT UI
# ==========================================
st.markdown('<p class="main-title">📈 Prediksi Harga Saham BBCA Menggunakan LSTM</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Aplikasi ini memprediksi harga saham BBCA berdasarkan model LSTM yang telah dilatih.</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙ Pengaturan Prediksi")
n_days = st.sidebar.slider("Prediksi berapa hari ke depan?", 1, 30, 7)


# Load data langsung dari repository
df = pd.read_csv(DATA_PATH)

if "Close" not in df.columns:
    st.error("File bbca.csv harus memiliki kolom 'Close'.")
    st.stop()


# ==========================================
# TOMBOL PREDIKSI
# ==========================================
if st.button("🔍 Jalankan Prediksi"):
    preds = predict_lstm(df, n_days)

    # =====================
    # TAMPILKAN CARD HASIL
    # =====================
    st.subheader("Hasil Prediksi LSTM")

    cols = st.columns(3)
    for i, p in enumerate(preds, start=1):
        with cols[(i-1) % 3]:
            st.markdown(
                f"""
                <div class='card'>
                    <h4>Hari ke-{i}</h4>
                    <h3 style='color:#1F4E79;'>Rp {p:,.2f}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

    # =====================
    # GRAFIK
    # =====================
    st.subheader("Grafik Aktual vs Prediksi")

    fig, ax = plt.subplots(figsize=(11,5))

    actual = df["Close"].astype(float).values
    future_index = np.arange(len(actual), len(actual) + n_days)

    ax.plot(actual, label="Harga Aktual", linewidth=2)
    ax.plot(future_index, preds, label="Prediksi", linestyle="--", marker="o", color="red")

    ax.set_title("Prediksi Harga Saham BBCA (LSTM)", fontsize=14)
    ax.set_xlabel("Index")
    ax.set_ylabel("Harga (IDR)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # =====================
    # TABEL PREDIKSI
    # =====================
    pred_df = pd.DataFrame({
        "Hari Ke": np.arange(1, n_days+1),
        "Prediksi Harga (IDR)": preds
    })

    st.subheader("Tabel Hasil Prediksi")
    st.dataframe(pred_df)
