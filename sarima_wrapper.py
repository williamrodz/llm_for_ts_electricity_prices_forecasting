# S-ARIMA prediction model
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
import utils
import matplotlib.pyplot as plt

def sarima_predict(df, column, training_start_index,training_end_index, prediction_length, plot=True):
  # Assuming df is your DataFrame and it has been previously defined

  training_df = df[training_start_index:training_end_index]
  forecast_index = df.index[training_end_index:training_end_index + prediction_length]

  # Determine seasonality 
  if column == 'Daily average':
    seasonal_period = 1
  elif column == 'Price_Ex_VAT':
    seasonal_period = 48
  else:
    raise ValueError("Unrecognized column name, cannont provide seasonal period")


  # print("SARIMA DEBUG")
  # print(f"training_df is\n{training_df}")
  # Use auto_arima to find the optimal p, d, q, P, D, Q, m values
  stepwise_fit = pm.auto_arima(training_df[column],
                              start_p=0, start_q=0, max_p=5, max_q=5,
                              start_P=0, start_Q=0, max_P=5, max_Q=5,
                              seasonal=True, m=seasonal_period,
                              trace=False,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)
  # Extract optimal parameters
  optimal_order = stepwise_fit.order
  optimal_seasonal_order = stepwise_fit.seasonal_order
  p, d, q = optimal_order
  P, D, Q, seasonal_period = optimal_seasonal_order

  # print(f'Optimal order: p={p}, d={d}, q={q}')
  # print(f'Optimal seasonal order: P={P}, D={D}, Q={Q}, m={m}')

  # Fit the SARIMA model with the optimal order
  model = sm.tsa.statespace.SARIMAX(training_df[column],
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, seasonal_period))
  model_fit = model.fit(disp=0)

  # Forecast
  comparison_values = df[forecast_index[0]:forecast_index[-1]][column]
  sarima_forecast = model_fit.forecast(steps=prediction_length)

  # Plot the original data and the forecast
  if plot:
    # plot a window of actual data two prediction windows to each side
    n = len(df)
    left_most_index = max(0, training_start_index - 2 * prediction_length)
    right_most_index = min(n, training_end_index + 2 * prediction_length)

    plt.figure(figsize=(8, 4))
    plt.title("S-ARIMA Forecast")
    plt.plot(df[column][left_most_index: training_start_index], color="royalblue", label="historical data")
    plt.plot(df[column][training_start_index: training_end_index], color="green", label="context")
    plt.plot(df[column][forecast_index[0]:right_most_index], color="royalblue", label="Post Context Historical data")
    plt.plot(forecast_index, sarima_forecast, color="tomato", label="median forecast")
    plt.legend()
    plt.show()

  return sarima_forecast
