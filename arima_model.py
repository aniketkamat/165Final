import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

data = yf.download("AAPL", period="1y", interval="1d")
close_prices = data["Close"]
arima_model = ARIMA(close_prices, order=(20, 1, 5))
fitted_model = arima_model.fit()
last_date = close_prices.index[-1]
target_date = pd.Timestamp("2025-07-01")
forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), end=target_date)
forecast_steps = len(forecast_dates)
forecast = fitted_model.forecast(steps=forecast_steps)
forecast.index = forecast_dates
forecast_result = fitted_model.get_forecast(steps=forecast_steps)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int(alpha=0.2)
if target_date in forecast.index:
    price_on_july1 = forecast.loc[target_date]
    print(price_on_july1)
print(forecast_ci)
plt.plot(close_prices.index, close_prices, label="Historical")
plt.plot(forecast.index, forecast, label="Forecast", color='blue')
plt.fill_between(forecast.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='lightblue', alpha=0.4, label="80% CI")
plt.axvline(x=target_date, color='red', label="July 1, 2025")
plt.scatter(target_date, price_on_july1, color='red', zorder=5)
plt.title("AAPL Forecast to July 1, 2025")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()