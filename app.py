import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- 1) Cache the data load so it never reloads on every rerun ---
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df

# --- 2) Cache & memoize the model fit to avoid retraining unless the data or orders change ---
@st.cache_resource
def fit_sarimax(series: pd.Series,
                order: tuple,
                seasonal_order: tuple,
                maxiter: int = 50) -> SARIMAX:
    """Fit SARIMAX with limited iterations for speed."""
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False, maxiter=maxiter)

# --- Load once ---
df = load_data("chocolate_sales.csv")

# --- Streamlit UI ---
st.title("Chocolate Sales Forecast (Fast SARIMAX)")
st.write("### Historical Weekly Sales")
st.line_chart(df["sales"])

# --- Split ---
train = df.iloc[:-52]
test  = df.iloc[-52:]

# --- Fit (cached) ---
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 52)
with st.spinner("Training SARIMAX…"):
    fit = fit_sarimax(train["sales"], order, seasonal_order)

# --- Forecast ---
fcast_res = fit.get_forecast(steps=52)
forecast   = fcast_res.predicted_mean
conf_int   = fcast_res.conf_int()

# --- Metrics ---
r2   = r2_score(test["sales"], forecast)
mse  = mean_squared_error(test["sales"], forecast)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(test["sales"], forecast)
mape = np.mean(np.abs((test["sales"] - forecast) / test["sales"])) * 100

# --- Plot ---
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

# --- Display Metrics ---
st.write("### Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("R²",   f"{r2:.3f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric("MAE",  f"{mae:.2f}")
c4.metric("MAPE", f"{mape:.2f}%")
