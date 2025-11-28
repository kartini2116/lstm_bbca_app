# ===========================
#   DEPLOY PREDIKSI BBCA
#   LSTM - Predict 23 Hari
# ===========================

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# ---------------------------
# 1. Load Model & Scaler
# ---------------------------

# LOAD MODEL .H5 TANPA ERROR
model = load_model("model_bbca.h5", compile=False)
model.compile(optimizer="adam", loss=MeanSquaredError())

# load scaler
scaler = joblib.load("scaler_bbca.pkl")

# ---------------------------
# 2. Load Data Close Price
# ---------------------------
df = pd.read_csv("bbca.csv")
close_prices = df[['Close']].astype(float)

# IMPORTANT: jangan scaler.fit() ulang
scaled_close = scaler.transform(close_prices)

# ---------------------------
# 3. Fungsi Prediksi Future
# ---------------------------
def predict_future(model, scaler, data, days=23, timestep=60):
    results_scaled = []
    window = data[-timestep:].reshape(1, timestep, 1)

    for _ in range(days):
        pred = model.predict(window, verbose=0)
        results_scaled.append(pred[0][0])

        # update window
        window = np.append(window[:, 1:, :], [[pred]], axis=1)

    # balikkan skala
    results = scaler.inverse_transform(np.array(results_scaled).reshape(-1,1))

    return results


# ---------------------------
# 4. Prediksi 23 Hari ke Depan
# ---------------------------
future_days = 23
future_pred = predict_future(model, scaler, scaled_close, days=future_days, timestep=60)

print("\n=== Prediksi 23 Hari Kedepan ===")
for i, p in enumerate(future_pred, start=1):
    print(f"Hari ke-{i}: Rp {p[0]:,.2f}")


# ---------------------------
# 5. Plot Grafik Harga Aktual + Prediksi Future
# ---------------------------
actual_prices = scaler.inverse_transform(scaled_close)

plt.figure(figsize=(12,5))
plt.plot(actual_prices, label="Harga Aktual")

start = len(actual_prices)
end = start + len(future_pred)

plt.plot(range(start, end), future_pred, 'o-', label="Prediksi Future")

plt.title("Prediksi Harga Saham BBCA Menggunakan LSTM")
plt.xlabel("Index Waktu")
plt.ylabel("Harga (IDR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
