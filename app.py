import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats

# ------------------------------ Load Data ------------------------------
df = pd.read_csv("chocolate_sales.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# ------------------------------ App Title ------------------------------
st.title("Chocolate Sales Forecast (Optimized ARIMA)")

# ------------------------------ Train/Test Split and Fit ------------------------------
train = df.iloc[:-52]
test = df.iloc[-52:]
order = (2, 0, 2)

with st.spinner(f"Training ARIMA{order}..."):
    model = ARIMA(train["sales"], order=order)
    model_fit = model.fit()

# ------------------------------ Forecasts ------------------------------
forecast_result = model_fit.get_forecast(steps=52, alpha=0.10)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()
forecast_dates = pd.date_range(start="2025-01-05", periods=52, freq='W-SUN')
forecast.index = forecast_dates
conf_int.index = forecast_dates
forecast_rounded = forecast.round(2)
conf_int_rounded = conf_int.round(2)

# ------------------------------ Tabs ------------------------------
tabs = st.tabs([
    "2025 Forecast & Summary",
    "2024 Model Evaluation",
    "Residual Diagnostics",
    "Historical Sales Lookup"
])

# ------------------------------ Tab 1: 2025 Forecast, Lookup, Summary, Download ------------------------------
with tabs[0]:
    st.subheader("Forecasted Chocolate Sales for 2025")

    # Forecast Plot
    fig_forecast, ax_forecast = plt.subplots(figsize=(10, 4))
    ax_forecast.plot(forecast_rounded.index, forecast_rounded, label="Forecasted Sales", color="blue")
    ax_forecast.fill_between(
        forecast_rounded.index,
        conf_int_rounded.iloc[:, 0],
        conf_int_rounded.iloc[:, 1],
        alpha=0.2,
        label="90% Confidence Interval"
    )
    ax_forecast.set_title("Projected Chocolate Sales (2025)")
    ax_forecast.set_ylabel("Sales ($)")
    ax_forecast.set_xlabel("Week")
    ax_forecast.legend()
    plt.tight_layout()
    st.pyplot(fig_forecast)

    # Week Selector
    st.subheader("Select a Week in 2025")
    selected_date = st.date_input(
        "Choose a forecast week:",
        min_value=forecast_rounded.index.min().date(),
        max_value=forecast_rounded.index.max().date(),
        value=forecast_rounded.index.min().date(),
        key="forecast_date"
    )
    selected_date = pd.to_datetime(selected_date)

    if selected_date not in forecast_rounded.index:
        st.warning("Please select a valid forecast week in 2025.")
    else:
        selected_forecast = forecast_rounded[selected_date]
        selected_ci = conf_int_rounded.loc[selected_date]
        st.metric("Forecasted Sales", f"{selected_forecast:.2f}")
        st.write(f"90% Confidence Interval: **[{selected_ci[0]:.2f}, {selected_ci[1]:.2f}]**")

    # Summary Metrics
    st.subheader("2025 Forecast Summary")
    total_sales = forecast_rounded.sum()
    avg_sales = forecast_rounded.mean()
    min_sales = forecast_rounded.min()
    max_sales = forecast_rounded.max()
    min_week = forecast_rounded.idxmin().date()
    max_week = forecast_rounded.idxmax().date()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Forecast Sales", f"{total_sales:,.2f}")
    col2.metric("Average Weekly Sales", f"{avg_sales:.2f}")
    col3.metric("Min Weekly Sales", f"{min_sales:.2f}", f"Week of {min_week}")
    col4.metric("Max Weekly Sales", f"{max_sales:.2f}", f"Week of {max_week}")

    # Download CSV
    download_df = pd.DataFrame({
        "date": forecast_rounded.index,
        "forecasted_sales": forecast_rounded.values,
        "ci_lower_90": conf_int_rounded.iloc[:, 0].values,
        "ci_upper_90": conf_int_rounded.iloc[:, 1].values
    }).set_index("date")

    csv = download_df.to_csv().encode('utf-8')
    st.download_button("Download 2025 Forecast as CSV", csv, "chocolate_sales_forecast_2025.csv", "text/csv")

# ------------------------------ Tab 2: 2024 Evaluation ------------------------------
with tabs[1]:
    st.subheader("Model Performance on 2024 Actual Data")
    test_forecast_result = model_fit.get_forecast(steps=52, alpha=0.10)
    test_forecast = test_forecast_result.predicted_mean
    test_conf_int = test_forecast_result.conf_int()
    test_forecast.index = test.index
    test_forecast_rounded = test_forecast.round(2)
    test_conf_int_rounded = test_conf_int.round(2)

    r2 = r2_score(test["sales"], test_forecast_rounded)
    mse = mean_squared_error(test["sales"], test_forecast_rounded)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test["sales"], test_forecast_rounded)
    mape = np.mean(np.abs((test["sales"] - test_forecast_rounded) / test["sales"])) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{r2:.3f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("MAE", f"{mae:.2f}")
    col4.metric("MAPE", f"{mape:.2f}%")

    fig_eval, ax_eval = plt.subplots(figsize=(10, 4))
    ax_eval.plot(test.index, test["sales"], label="Actual Sales", color="black")
    ax_eval.plot(test_forecast_rounded.index, test_forecast_rounded, label="Forecasted Sales", color="blue")
    ax_eval.fill_between(
        test_forecast_rounded.index,
        test_conf_int_rounded.iloc[:, 0],
        test_conf_int_rounded.iloc[:, 1],
        alpha=0.2,
        label="90% Confidence Interval"
    )
    ax_eval.set_title("ARIMA Forecast vs Actual (2024)")
    ax_eval.set_ylabel("Sales ($)")
    ax_eval.set_xlabel("Week")
    ax_eval.legend()
    plt.tight_layout()
    st.pyplot(fig_eval)

# ------------------------------ Tab 3: Residual Diagnostics ------------------------------
with tabs[2]:
    st.subheader("Residual Diagnostics")
    residuals = model_fit.resid

    fig_resid, ax_resid = plt.subplots(figsize=(10, 4))
    ax_resid.plot(residuals)
    ax_resid.set_title("Residuals Over Time")
    ax_resid.set_ylabel("Residual")
    ax_resid.grid(True)
    plt.tight_layout()
    st.pyplot(fig_resid)

    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
    ax_hist.hist(residuals, bins=20, edgecolor="k", alpha=0.7)
    ax_hist.set_title("Histogram of Residuals")
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig_hist)

    fig_qq, ax_qq = plt.subplots(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=ax_qq)
    ax_qq.set_title("Q-Q Plot of Residuals")
    plt.tight_layout()
    st.pyplot(fig_qq)

    fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
    plot_acf(residuals, ax=ax_acf, lags=40)
    ax_acf.set_title("Autocorrelation (ACF) of Residuals")
    plt.tight_layout()
    st.pyplot(fig_acf)

# ------------------------------ Tab 4: Historical Sales Lookup ------------------------------
with tabs[3]:
    st.subheader("Historical Weekly Sales")
    st.line_chart(df["sales"])

    st.subheader("Look Up Actual Sales for a Past Week")
    historical_date = st.date_input(
        "Choose a date to view historical sales:",
        min_value=df.index.min().date(),
        max_value=df.index.max().date(),
        value=df.index[-1].date(),
        key="history_date"
    )
    historical_date = pd.to_datetime(historical_date)

    if historical_date not in df.index:
        st.warning("Selected date is not in the dataset.")
    else:
        actual_sales = df.loc[historical_date, "sales"]
        st.metric("Actual Sales", f"{actual_sales:.2f}")
