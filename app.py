import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---- Load Data ----
df = pd.read_csv("chocolate_sales.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# ---- Streamlit App ----
st.title("Chocolate Sales Forecast (SARIMAX)")
st.write("### Historical Weekly Sales")
st.line_chart(df["sales"])

# ---- Train/Test Split ----
train = df.iloc[:-52]   # first ~2 years
test  = df.iloc[-52:]   # last year

# ---- Fit Seasonal ARIMA (SARIMAX) ----
# non‑seasonal order (p, d, q) = (1, 1, 1)
# seasonal order  (P, D, Q, s) = (1, 1, 1, 52)
with st.spinner("Training SARIMAX(1,1,1)x(1,1,1,52)…"):
    model = SARIMAX(
        train["sales"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)

# ---- Forecast 52 Weeks ----
forecast_res = fit.get_forecast(steps=52)
forecast     = forecast_res.predicted_mean
conf_int     = forecast_res.conf_int()

# ---- Compute Metrics ----
r2   = r2_score(test["sales"], forecast)
mse  = mean_squared_error(test["sales"], forecast)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(test["sales"], forecast)
mape = np.mean(np.abs((test["sales"] - forecast) / test["sales"])) * 100

# ---- Plot Forecast vs Actual ----
st.write("### Forecast vs Actual (Last 52 Weeks)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(test.index, test["sales"], label="Actual", color="black")
ax.plot(test.index, forecast,     label="Forecast", color="blue")
ax.fill_between(
    test.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    alpha=0.2,
    label="95% CI"
)
ax.set_title("SARIMAX Forecast (52 Weeks)")
ax.legend()
st.pyplot(fig)

# ---- Display Metrics ----
st.write("### Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("R²",   f"{r2:.3f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric("MAE",  f"{mae:.2f}")
c4.metric("MAPE", f"{mape:.2f}%")

