import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats

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
order = (3, 1, 2)  # Replace with best order from auto_arima if needed

with st.spinner(f"Training ARIMA{order}..."):
    model = ARIMA(train["sales"], order=order)
    model_fit = model.fit()

# ---- Forecast for NEXT YEAR (2025) ----
forecast_result = model_fit.get_forecast(steps=52, alpha=0.10)  # 90% CI
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# Set forecast index to all Sundays in 2025
forecast_dates = pd.date_range(start="2025-01-05", periods=52, freq='W-SUN')
forecast.index = forecast_dates
conf_int.index = forecast_dates

# ---- Forecast for TEST PERIOD (last 52 weeks in dataset, 2024) ----
test_forecast_result = model_fit.get_forecast(steps=52, alpha=0.10)
test_forecast = test_forecast_result.predicted_mean
test_conf_int = test_forecast_result.conf_int()
test_forecast.index = test.index

# Round forecast and confidence intervals to 2 decimals
forecast_rounded = forecast.round(2)
conf_int_rounded = conf_int.round(2)
test_forecast_rounded = test_forecast.round(2)
test_conf_int_rounded = test_conf_int.round(2)

# ---- Calculate metrics on TEST PERIOD forecast ----
r2   = r2_score(test["sales"], test_forecast_rounded)
mse  = mean_squared_error(test["sales"], test_forecast_rounded)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(test["sales"], test_forecast_rounded)
mape = np.mean(np.abs((test["sales"] - test_forecast_rounded) / test["sales"])) * 100

# ---- Next Year Forecast Summary (2025) ----
total_next_year_sales = forecast_rounded.sum()
average_weekly_sales = forecast_rounded.mean()
min_week_sales = forecast_rounded.min()
min_week_date = forecast_rounded.idxmin().date()
max_week_sales = forecast_rounded.max()
max_week_date = forecast_rounded.idxmax().date()

st.write("## Next Year Forecast Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Forecast Sales (Next 52 Weeks)", f"{total_next_year_sales:,.2f}")
col2.metric("Average Weekly Sales", f"{average_weekly_sales:,.2f}")
col3.metric("Min Weekly Sales", f"{min_week_sales:,.2f}", f"Week of {min_week_date}")
col4.metric("Max Weekly Sales", f"{max_week_sales:,.2f}", f"Week of {max_week_date}")

# ---- Date Picker for Next Year Forecast (2025) ----
st.write("### Select a Date from Next Year Forecast")
selected_date = st.date_input(
    "Pick a forecast date:",
    min_value=forecast_rounded.index.min().date(),
    max_value=forecast_rounded.index.max().date(),
    value=forecast_rounded.index.min().date()
)

selected_date = pd.to_datetime(selected_date)

if selected_date not in forecast_rounded.index:
    st.error("Selected date is out of forecast range. Please select a valid date.")
    selected_date_for_plot = forecast_rounded.index.min()
else:
    selected_date_for_plot = selected_date
    selected_forecast = forecast_rounded[selected_date]
    selected_ci = conf_int_rounded.loc[selected_date]

    st.write(f"#### Forecast for {selected_date.date()}:")
    st.metric("Forecasted Sales", f"{selected_forecast:.2f}")
    st.write(f"90% Confidence Interval: **[{selected_ci[0]:.2f}, {selected_ci[1]:.2f}]**")

# ---- Download Button ----
download_df = pd.DataFrame({
    "date": forecast_rounded.index,
    "forecasted_sales": forecast_rounded.values,
    "ci_lower_90": conf_int_rounded.iloc[:, 0].values,
    "ci_upper_90": conf_int_rounded.iloc[:, 1].values
})
download_df.set_index("date", inplace=True)

csv = download_df.to_csv().encode('utf-8')

st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name="chocolate_sales_forecast.csv",
    mime="text/csv"
)

# ---- Residual Diagnostics ----
st.write("### Residual Diagnostics")

residuals = model_fit.resid

fig_resid, ax_resid = plt.subplots(figsize=(10, 3))
ax_resid.plot(residuals)
ax_resid.set_title("Residuals Over Time")
ax_resid.set_ylabel("Residual")
ax_resid.grid(True)
st.pyplot(fig_resid)

fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
ax_hist.hist(residuals, bins=20, edgecolor="k", alpha=0.7)
ax_hist.set_title("Histogram of Residuals")
ax_hist.set_xlabel("Residual")
ax_hist.set_ylabel("Frequency")
st.pyplot(fig_hist)

fig_qq, ax_qq = plt.subplots(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=ax_qq)
ax_qq.set_title("Q-Q Plot of Residuals")
st.pyplot(fig_qq)

fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
plot_acf(residuals, ax=ax_acf, lags=40)
ax_acf.set_title("ACF Plot of Residuals")
st.pyplot(fig_acf)

# ---- Plot Forecast vs Actual for TEST PERIOD (2024) ----
st.write("### Forecast vs Actual (Last 52 Weeks)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(test.index, test["sales"], label="Actual", color="black")
ax.plot(test_forecast_rounded.index, test_forecast_rounded, label="ARIMA Forecast", color="blue")
ax.fill_between(
    test_forecast_rounded.index,
    test_conf_int_rounded.iloc[:, 0],
    test_conf_int_rounded.iloc[:, 1],
    alpha=0.2,
    label="90% CI"
)
ax.axvline(x=selected_date_for_plot, color="red", linestyle="--", label=f"Selected Date")
ax.set_title("ARIMA Forecast (Last 52 Weeks - Test Period)")
ax.legend()
st.pyplot(fig)

# ---- Display Overall Metrics for TEST PERIOD ----
st.write("### Model Performance on Last 52 Weeks of Actual Data")
col1, col2, col3, col4 = st.columns(4)
col1.metric("R²",   f"{r2:.3f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("MAE",  f"{mae:.2f}")
col4.metric("MAPE", f"{mape:.2f}%")

