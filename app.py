import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load data
df = pd.read_csv("chocolate_sales.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

st.title("Chocolate Sales Forecast")
st.write("### Historical Weekly Sales")
st.line_chart(df["sales"])

# Train-test split: last 52 weeks as test
train = df.iloc[:-52]
test = df.iloc[-52:]

# Fit ARIMA model (manually chosen order)
order = (2, 1, 2)  # Adjust this if needed
with st.spinner(f"Training ARIMA model with order {order}..."):
    model = ARIMA(train["sales"], order=order)
    model_fit = model.fit()

# Forecast next 52 weeks
forecast_result = model_fit.get_forecast(steps=52)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# Compute accuracy metrics
r2 = r2_score(test["sales"], forecast)
rmse = mean_squared_error(test["sales"], forecast, squared=False)
mae = mean_absolute_error(test["sales"], forecast)
mape = np.mean(np.abs((test["sales"] - forecast) / test["sales"])) * 100

# Plot forecast vs actual
st.write("### Forecast vs Actual (Last 52 Weeks)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(test.index, test["sales"], label="Actual", color="black")
ax.plot(test.index, forecast, label="Forecast", color="blue")
ax.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='blue', alpha=0.2, label="95% CI")
ax.set_title("ARIMA Forecast (52 weeks)")
ax.legend()
st.pyplot(fig)

# Display metrics
st.write("### Model Performance")
st.metric("R²", f"{r2:.3f}")
st.metric("RMSE", f"{rmse:.2f}")
st.metric("MAE", f"{mae:.2f}")
st.metric("MAPE", f"{mape:.2f}%")
