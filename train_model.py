import pandas as pd
import numpy as np
import json
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load data
df = pd.read_csv("chocolate_sales.csv", parse_dates=["date"])
df.set_index("date", inplace=True)
train, test = df.iloc[:-52], df.iloc[-52:]

# Fit SARIMAX(1,1,1)x(1,1,1,52)
model = SARIMAX(
    train["sales"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)
fit = model.fit(disp=False, method="css-mle", maxiter=10, tol=1e-3)

# Forecast + CI
fc_res = fit.get_forecast(steps=52)
forecast = fc_res.predicted_mean.rename("forecast")
ci = fc_res.conf_int()
ci.columns = ["lower_ci", "upper_ci"]

# Save forecast & CI
out = pd.concat([forecast, ci], axis=1)
out.index.name = "date"
out.to_csv("precomputed/forecast.csv")

# Compute metrics
r2   = r2_score(test["sales"], forecast)
mse  = mean_squared_error(test["sales"], forecast)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(test["sales"], forecast)
mape = np.mean(np.abs((test["sales"] - forecast) / test["sales"])) * 100

metrics = {
    "R2":   round(r2,   3),
    "RMSE": round(rmse, 2),
    "MAE":  round(mae,  2),
    "MAPE": round(mape, 2)
}
with open("precomputed/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
