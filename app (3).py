import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# ============================================
# CONFIG APLIKASI
# ============================================
st.set_page_config(
    page_title="Prediksi Saham BBCA - LSTM",
    layout="wide",
    page_icon="📈"
)

MODEL_PATH = "model_bbca.keras"
SCALER_PATH = "scaler_bbca.pkl"
DATA_PATH = "bbca.csv"
TIMESTEP = 60

# ============================================
# LOAD MODEL & SCALER
# ============================================
@st.cache_resource
def load_lstm_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

model = load_lstm_model()
scaler: MinMaxScaler = load_scaler()

# ============================================
# FUNGSI PREDIKSI
# ============================================
def predict_future(df, n_future):
    close_prices = df[['Close']].astype(float).values
    scaled = scaler.transform(close_prices)

    last_seq = scaled[-TIMESTEP:]
    seq = last_seq.reshape(1, TIMESTEP, 1)

    preds_real = []

    for _ in range(n_future):
        pred_scaled = model.predict(seq, verbose=0)[0][0]
        pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
        preds_real.append(pred_real)

        new_seq = np.append(seq.flatten()[1:], pred_scaled)
        seq = new_seq.reshape(1, TIMESTEP, 1)

    return preds_real

# ============================================
# SIDEBAR NAVIGASI
# ============================================
menu = st.sidebar.radio(
    "📌 Navigasi",
    ["🏠 Home", "🔮 Prediksi Harga"]
)

# ============================================
# HALAMAN HOME
# ============================================
if menu == "🏠 Home":
    st.title("📈 Prediksi Harga Saham BBCA Menggunakan LSTM")
    st.write("""
Selamat datang di aplikasi **Prediksi Harga Saham BBCA** berbasis **Long Short-Term Memory (LSTM)**.

Aplikasi ini:
- Menggunakan model LSTM yang dilatih dari data historis BBCA  
- Membaca **data otomatis** dari repository (bbca.csv)  
- Menyediakan **prediksi jumlah hari ke depan**  
- Menampilkan **grafik aktual vs prediksi**  
- Menyediakan UI profesional dan mudah digunakan  

Klik menu **'🔮 Prediksi Harga'** untuk mulai.
    """)

    st.info("Data otomatis di-load dari file `bbca.csv` di repository Anda.")

# ============================================
# HALAMAN PREDIKSI
# ============================================
elif menu == "🔮 Prediksi Harga":

    st.title("🔮 Prediksi Harga Saham BBCA")

    df = pd.read_csv(DATA_PATH)

    if "Close" not in df.columns:
        st.error("File `bbca.csv` harus memiliki kolom 'Close'.")
        st.stop()

    st.sidebar.header("⚙ Pengaturan Prediksi")
    n_days = st.sidebar.slider("Prediksi berapa hari ke depan?", 1, 60, 7)

    if st.button("🚀 Jalankan Prediksi"):
        preds = predict_future(df, n_days)
        st.success("Prediksi berhasil dibuat!")

        # =============================
        # Hasil Prediksi
        # =============================
        st.subheader("📌 Hasil Prediksi Harga BBCA")
        for i, p in enumerate(preds, start=1):
            st.write(f"**Hari ke-{i}: Rp {p:,.2f}**")

        # =============================
        # Grafik Interaktif
        # =============================
        st.subheader("📈 Grafik Harga Aktual vs Prediksi")

        actual = df["Close"].values
        future_index = np.arange(len(actual), len(actual) + n_days)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=actual, mode='lines', name='Aktual', line=dict(width=3)
        ))
        fig.add_trace(go.Scatter(
            x=future_index, y=preds,
            mode='lines+markers', name='Prediksi',
            line=dict(width=3, dash='dash')
        ))

        fig.update_layout(
            title="Prediksi Harga BBCA (LSTM)",
            xaxis_title="Index Waktu",
            yaxis_title="Harga (IDR)",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # =============================
        # Tabel Prediksi
        # =============================
        pred_df = pd.DataFrame({
            "Hari Ke": np.arange(1, n_days + 1),
            "Prediksi Harga": preds
        })

        st.subheader("📄 Tabel Prediksi")
        st.dataframe(pred_df, use_container_width=True)
