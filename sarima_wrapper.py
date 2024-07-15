# S-ARIMA prediction model
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
import utils
import matplotlib.pyplot as plt


DATE_COLUMN = "Date"

def sarima_predict(df, column, training_start_index,training_end_index, prediction_length, plot=True):
  # Assuming df is your DataFrame and it has been previously defined
  training_df = None

  if type(training_start_index) == str:
    # convert dates to index
    mask = (df[DATE_COLUMN] >= training_start_index) & (df[DATE_COLUMN] <= training_end_index)
    training_df = df.loc[mask]
    training_start_index = utils.find_first_occurrence_index(df, training_start_index,DATE_COLUMN)
    training_end_index = utils.find_first_occurrence_index(df, training_end_index,DATE_COLUMN)
    forecast_index = range(training_end_index + 1,training_end_index + 1 + prediction_length)
  elif not training_start_index is None and not training_end_index is None:
    training_df = df[training_start_index:training_end_index]
    forecast_index = df.index[training_end_index:training_end_index + prediction_length]
  else:
    raise ValueError(f'training_start_index is {type(training_start_index)}. Must be int or str.')

  # print("SARIMA DEBUG")
  # print(f"training_df is\n{training_df}")
  # Use auto_arima to find the optimal p, d, q, P, D, Q, m values
  stepwise_fit = pm.auto_arima(training_df[column],
                              start_p=0, start_q=0, max_p=5, max_q=5,
                              start_P=0, start_Q=0, max_P=5, max_Q=5,
                              seasonal=True, m=12,  # Assuming monthly seasonal data, adjust m as needed
                              trace=False,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)
  # Extract optimal parameters
  optimal_order = stepwise_fit.order
  optimal_seasonal_order = stepwise_fit.seasonal_order
  p, d, q = optimal_order
  P, D, Q, m = optimal_seasonal_order

  # print(f'Optimal order: p={p}, d={d}, q={q}')
  # print(f'Optimal seasonal order: P={P}, D={D}, Q={Q}, m={m}')

  # Fit the SARIMA model with the optimal order
  model = sm.tsa.statespace.SARIMAX(training_df[column],
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, m))
  model_fit = model.fit(disp=0)
  # Forecast

  comparison_values = df[forecast_index[0]:forecast_index[-1]][column]

  sarima_forecast = model_fit.forecast(steps=prediction_length)

  # Plot the original data and the forecast
  if plot:
    plt.figure(figsize=(8, 4))
    plt.title("S-ARIMA Forecast")
    plt.plot(df[column][: training_start_index], color="royalblue", label="historical data")
    plt.plot(df[column][training_start_index: training_end_index], color="green", label="historical data")
    plt.plot(forecast_index, sarima_forecast, color="tomato", label="median forecast")
    plt.plot(df[column][forecast_index[0]:], color="royalblue", label="historical data")
    plt.legend()
    plt.show()

  return sarima_forecast
