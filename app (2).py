import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from utils import load_lstm_model, predict_future

# ================================
#   CONFIGURASI DASHBOARD
# ================================
st.set_page_config(
    page_title="LSTM BBCA Stock Prediction",
    page_icon="📈",
    layout="wide"
)

# CSS styling untuk tampilan profesional
st.markdown("""
<style>
.big-font {
    font-size:28px !important;
    font-weight: bold;
}
.card {
    padding: 20px;
    background-color: #1f2937;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-size: 20px;
}
.metric-title {
    font-size: 16px;
    color: #9ca3af;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ===========================================
#      SIDEBAR
# ===========================================
st.sidebar.title("📊 Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["Dashboard", "Prediksi 3 Hari"]
)

# Load model
model = load_lstm_model("model_lstm_bbca")


# ================================================================
#                      DASHBOARD UTAMA
# ================================================================
if menu == "Dashboard":
    st.title("📈 Dashboard Prediksi Saham BBCA Menggunakan LSTM")

    upload = st.file_uploader("Upload file bbca.csv", type=["csv"])

    if upload is not None:
        df = pd.read_csv(upload)
        st.dataframe(df.tail())

        close_prices = df["Close"].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_prices)

        # Residual hanya ditampilkan jika model berisi prediksi valid
        st.subheader("📉 Grafik Harga Saham")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df["Close"],
            mode='lines',
            name='Harga Close',
            line=dict(width=2)
        ))
        fig.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Index",
            yaxis_title="Harga"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Silakan upload file **bbca.csv** untuk memulai.")


# ================================================================
#               PREDIKSI 3 HARI KE DEPAN
# ================================================================
if menu == "Prediksi 3 Hari":

    st.title("🔮 Prediksi Harga BBCA 3 Hari Kedepan")

    upload = st.file_uploader("Upload file bbca.csv", type=["csv"])

    if upload is not None:

        df = pd.read_csv(upload)

        close_price = df["Close"].values.reshape(-1,1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close_price)

        window = 60
        last_sequence = scaled[-window:]

        with st.spinner("Model sedang memprediksi..."):
            pred_scaled = predict_future(model, last_sequence, n_future=3)
            pred = scaler.inverse_transform(pred_scaled.reshape(-1,1))

        # ===========================
        #   Tampilan Prediksi
        # ===========================
        st.header("📌 Hasil Prediksi 3 Hari Kedepan")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="card">
                <div class="metric-title">Prediksi Hari 1</div>
                <div class="metric-value">{pred[0][0]:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="card">
                <div class="metric-title">Prediksi Hari 2</div>
                <div class="metric-value">{pred[1][0]:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="card">
                <div class="metric-title">Prediksi Hari 3</div>
                <div class="metric-value">{pred[2][0]:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("📉 Grafik Prediksi vs Aktual")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df["Close"],
            mode='lines',
            name='Aktual',
            line=dict(width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[len(df), len(df)+1, len(df)+2],
            y=[pred[0][0], pred[1][0], pred[2][0]],
            mode='lines+markers',
            name='Prediksi',
            line=dict(width=3, dash='dot')
        ))

        fig.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Index",
            yaxis_title="Harga"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Silakan upload file **bbca.csv** untuk memulai prediksi.")
