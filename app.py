import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import json

# — Cache your data loads so edits don’t re‑read files
@st.cache_data
def load_history(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df

@st.cache_data
def load_forecast(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df

@st.cache_data
def load_metrics(path):
    return json.loads(open(path).read())

# — Load everything
history = load_history("chocolate_sales.csv")
forecast = load_forecast("precomputed/forecast.csv")
metrics  = load_metrics("precomputed/metrics.json")

# — App display
st.title("Chocolate Sales Forecast (Precomputed SARIMAX)")

st.write("### Historical Weekly Sales")
st.line_chart(history["sales"])

st.write("### 52‑Week Forecast vs Actual")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(history.index[-52:], history["sales"].iloc[-52:], label="Actual", color="black")
ax.plot(forecast.index, forecast["forecast"],       label="Forecast", color="blue")
ax.fill_between(
    forecast.index,
    forecast["lower_ci"],
    forecast["upper_ci"],
    alpha=0.2,
    label="95% CI")
ax.legend()
st.pyplot(fig)

st.write("### Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("R²",   metrics["R2"])
c2.metric("RMSE", metrics["RMSE"])
c3.metric("MAE",  metrics["MAE"])
c4.metric("MAPE", f"{metrics['MAPE']}%")
