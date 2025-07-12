import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---- Load Data ----
df = pd.read_csv("chocolate_sales.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# ---- Streamlit App ----
st.title("Chocolate Sales Forecast (Optimized ARIMA)")
st.write("### Historical Weekly Sales")
st.line_chart(df["sales"])

# ---- Train/Test Split ----
train = df.iloc[:-52]
test  = df.iloc[-52:]

# ---- Fit ARIMA Model ----
order = (3, 1, 2)  # ← Replace with best order from auto_arima

with st.spinner(f"Training ARIMA{order}..."):
    model = ARIMA(train["sales"], order=order)
    model_fit = model.fit()

# ---- Forecast 52 weeks ----
forecast_result = model_fit.get_forecast(steps=52)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()
forecast.index = test.index  # align forecast index to match actual dates
conf_int.index = test.index

# ---- User selects week ----
st.write("### Select a Week to View Forecast Details")
week_num = st.slider("Choose a week number (1 = next week):", 1, 52, 1)
selected_date = forecast.index[week_num - 1]
selected_forecast = forecast[selected_date]
selected_ci = conf_int.loc[selected_date]

st.write(f"#### Week {week_num} ({selected_date.date()}):")
st.metric("Forecasted Sales", f"{selected_forecast:.2f}")
st.write(f"95% Confidence Interval: **[{selected_ci[0]:.2f}, {selected_ci[1]:.2f}]**")

# ---- Plot Forecast vs Actual ----
st.write("### Forecast vs Actual (Last 52 Weeks)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(test.index, test["sales"], label="Actual", color="black")
ax.plot(test.index, forecast, label="Forecast", color="blue")
ax.fill_between(
    test.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    alpha=0.2,
    label="95% CI"
)
ax.axvline(x=selected_date, color="red", linestyle="--", label=f"Week {week_num}")
ax.set_title("ARIMA Forecast (52 Weeks)")
ax.legend()
st.pyplot(fig)

# ---- Display Overall Metrics ----
st.write("### Model Performance")
r2   = r2_score(test["sales"], forecast)
mse  = mean_squared_error(test["sales"], forecast)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(test["sales"], forecast)
mape = np.mean(np.abs((test["sales"] - forecast) / test["sales"])) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("R²",   f"{r2:.3f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("MAE",  f"{mae:.2f}")
col4.metric("MAPE", f"{mape:.2f}%")

