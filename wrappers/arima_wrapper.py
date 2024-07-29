import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
import statsmodels.api as sm

def arima_predict(df, column, training_start_index, training_end_index, prediction_length, plot=True):
    # Assuming df is your DataFrame and it has been previously defined

    training_df = df[training_start_index:training_end_index]
    forecast_index = df.index[training_end_index:training_end_index + prediction_length]

    # Use auto_arima to find the optimal p, d, q values for ARIMA
    stepwise_fit = pm.auto_arima(training_df[column],
                                 start_p=0, start_q=0, max_p=3, max_q=3,
                                 seasonal=False,
                                 trace=False,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)
    
    # Extract optimal parameters
    optimal_order = stepwise_fit.order
    p, d, q = optimal_order

    # Fit the ARIMA model with the optimal order
    model = sm.tsa.ARIMA(training_df[column], order=(p, d, q))
    model_fit = model.fit()

    # Forecast
    comparison_values = df[forecast_index[0]:forecast_index[-1]][column]
    arima_forecast = model_fit.forecast(steps=prediction_length)

    # Plot the original data and the forecast
    if plot:
        # plot a window of actual data two prediction windows to each side
        n = len(df)
        left_most_index = max(0, training_start_index - 2 * prediction_length)
        right_most_index = min(n, training_end_index + 2 * prediction_length)

        plt.figure(figsize=(8, 4))
        plt.title("ARIMA Forecast")
        plt.plot(df[column][left_most_index: training_start_index], color="royalblue", label="historical data")
        plt.plot(df[column][training_start_index: training_end_index], color="green", label="context")
        plt.plot(df[column][forecast_index[0]:right_most_index], color="royalblue", label="Post Context Historical data")
        plt.plot(forecast_index, arima_forecast, color="tomato", label="median forecast")
        plt.legend()
        plt.show()

    return arima_forecast

# Example usage:
# Assuming you have a DataFrame `df` with a time series column `column`
# arima_forecast = arima_predict(df, 'column_name', 0, 100, 10, plot=True)