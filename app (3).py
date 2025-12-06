import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# =========================================
# KONFIGURASI APLIKASI
# =========================================
st.set_page_config(page_title="Prediksi Saham BBCA", layout="wide")

MODEL_PATH = "model_bbca.keras"
SCALER_PATH = "scaler_bbca.pkl"
DATA_PATH   = "bbca.csv"
TIMESTEP = 60

# =========================================
# LOAD MODEL & SCALER
# =========================================
@st.cache_resource
def load_lstm_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

model = load_lstm_model()
scaler: MinMaxScaler = load_scaler()


# =========================================
# FUNGSI PREDIKSI
# =========================================
def predict_lstm(df, n_future):
    close_prices = df[['Close']].astype(float).values
    scaled = scaler.transform(close_prices)

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


# =========================================
# SIDEBAR MENU
# =========================================
menu = st.sidebar.radio(
    "📌 Navigasi",
    ["🏠 Home", "🔮 Prediksi LSTM", "📊 Visualisasi Chart"]
)

# =========================================
# HOME PAGE
# =========================================
if menu == "🏠 Home":
    st.title("📈 Prediksi Harga Saham BBCA Menggunakan LSTM")
    st.write("""
Aplikasi ini menggunakan model **LSTM (Long Short-Term Memory)** untuk memprediksi harga saham 
BBCA beberapa hari ke depan.

### Navigasi:
- 🔮 **Prediksi LSTM** — Prediksi harga n-hari ke depan  
- 📊 **Visualisasi Chart** — Grafik candlestick & tren harga interaktif  

Dataset default dibaca otomatis dari **repository (bbca.csv)**.
    """)

# =========================================
# HALAMAN PREDIKSI
# =========================================
elif menu == "🔮 Prediksi LSTM":

    st.title("🔮 Prediksi Harga Saham BBCA")

    st.sidebar.header("⚙ Pengaturan Prediksi")
    n_days = st.sidebar.slider("Prediksi berapa hari ke depan?", 1, 30, 7)

    uploaded_file = st.file_uploader("Upload CSV (opsional)", type=["csv"])

    df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(DATA_PATH)

    if "Close" not in df.columns:
        st.error("CSV harus memiliki kolom 'Close'.")
        st.stop()

    if st.button("🚀 Jalankan Prediksi"):
        preds = predict_lstm(df, n_days)
        st.success("Prediksi berhasil dibuat!")

        # =============================
        # Tampilkan hasil prediksi
        # =============================
        st.subheader("📌 Hasil Prediksi")
        for i, p in enumerate(preds, start=1):
            st.write(f"**Hari ke-{i}: Rp {p:,.2f}**")

        # =============================
        # Grafik Interaktif
        # =============================
        st.subheader("📈 Grafik Prediksi (Plotly Interaktif)")

        actual = df["Close"].values
        future_index = np.arange(len(actual), len(actual) + n_days)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=actual, mode='lines', name='Aktual',
            line=dict(width=3)
        ))

        fig.add_trace(go.Scatter(
            x=future_index, y=preds, mode='lines+markers',
            name='Prediksi', line=dict(dash='dash')
        ))

        fig.update_layout(
            title="Prediksi Harga BBCA",
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


# =========================================
# HALAMAN VISUALISASI
# =========================================
elif menu == "📊 Visualisasi Chart":

    st.title("📊 Visualisasi Saham BBCA (Interaktif)")

    uploaded_file = st.file_uploader("Upload CSV (opsional)", type=["csv"])

    df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(DATA_PATH)

    if "Close" not in df.columns:
        st.error("CSV harus memiliki kolom: Date, Open, High, Low, Close.")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'])

    # =============================
    # Candlestick Chart
    # =============================
    st.subheader("🕯️ Candlestick Chart")

    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    fig.update_layout(
        title="Candlestick Chart BBCA",
        xaxis_title="Tanggal",
        yaxis_title="Harga (IDR)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =============================
    # Line Chart Harga Close
    # =============================
    st.subheader("📈 Grafik Harga Close")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        mode="lines", name="Close Price"
    ))

    fig2.update_layout(
        title="Grafik Harga Close BBCA",
        xaxis_title="Tanggal",
        yaxis_title="Harga (IDR)",
        template="plotly_white"
    )

    st.plotly_chart(fig2, use_container_width=True)
