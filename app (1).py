import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

st.title("Prediksi Harga Saham BBCA - LSTM")

model = load_model("model_bbca.h5", compile=False)
model.compile(optimizer="adam", loss=MeanSquaredError())

scaler = joblib.load("scaler_bbca.pkl")

df = pd.read_csv("bbca.csv")
close_prices = df[['Close']].astype(float)
scaled_close = scaler.transform(close_prices)

def predict_future(model, scaler, data, days=23, timestep=60):
    results_scaled = []
    window = data[-timestep:].reshape(1, timestep, 1)

    for _ in range(days):
        pred = model.predict(window, verbose=0)
        pred_reshaped = pred.reshape(1, 1, 1)
        window = np.concatenate([window[:, 1:, :], pred_reshaped], axis=1)
        results_scaled.append(pred[0][0])

    results = scaler.inverse_transform(np.array(results_scaled).reshape(-1,1))
    return results

future_days = 23
future_pred = predict_future(model, scaler, scaled_close, days=future_days)

st.subheader("Prediksi 23 Hari Kedepan")
for i, p in enumerate(future_pred, start=1):
    st.write(f"Hari ke-{i}: Rp {p[0]:,.2f}")

actual_prices = scaler.inverse_transform(scaled_close)

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(actual_prices, label="Harga Aktual")
start = len(actual_prices)
end = start + len(future_pred)
ax.plot(range(start, end), future_pred, 'o-', label="Prediksi Future")
ax.legend()
st.pyplot(fig)
