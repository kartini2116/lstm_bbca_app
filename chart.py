import streamlit as st
import pandas as pd
import plotly.graph_objects as go

DATA_PATH = "bbca.csv"

st.title("📊 Visualisasi Saham BBCA (Interaktif)")

uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"])

df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(DATA_PATH)

if "Close" not in df.columns:
    st.error("CSV harus memiliki kolom: Date, Open, High, Low, Close.")
    st.stop()

# Convert date
df['Date'] = pd.to_datetime(df['Date'])

# Candlestick Chart
fig = go.Figure(data=[go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])

fig.update_layout(
    title="Candlestick Chart BBCA",
    xaxis_title="Tanggal",
    yaxis_title="Harga (IDR)",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
