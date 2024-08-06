import matplotlib.pyplot as plt
import pmdarima as pm
import statsmodels.api as sm

def arima_predict(df, column, training_start_index, training_end_index, prediction_length, plot=True):
    """
    input: 
        df: DataFrame
            Represents the time series data
        column: str
            Represents the column in the DataFrame to predict
        training_start_index: int
            Represents the start index of the training data
        training_end_index: int
            Represents the end index of the training data
        prediction_length: int
            Number of data points to predict
        plot: bool
            Determines whether to plot the forecast
    output:
        arima_forecast: [float]
            Returns the predicted values
        stderr: [float]
            Returns the standard errors of the forecast
    """
    # Extract training and forecast data
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

    # Forecast using the get_forecast method
    forecast_result = model_fit.get_forecast(steps=prediction_length)
    arima_forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)
    
    # Calculate standard errors
    stderr = forecast_result.se_mean

    # Plot the original data and the forecast
    if plot:
        # Plot a window of actual data two prediction windows to each side
        n = len(df)
        left_most_index = max(0, training_start_index - 2 * prediction_length)
        right_most_index = min(n, training_end_index + 2 * prediction_length)

        plt.figure(figsize=(10, 5))
        plt.title("ARIMA Forecast")
        plt.plot(df[column][left_most_index: training_start_index], color="royalblue", label="Historical Data")
        plt.plot(df[column][training_start_index: training_end_index], color="green", label="Context")
        plt.plot(df[column][forecast_index[0]:right_most_index], color="royalblue", label="Post Context Historical Data")
        plt.plot(forecast_index, arima_forecast, color="tomato", label="Median Forecast")
        plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='tomato', alpha=0.2, label="95% Confidence Interval")
        plt.legend()
        plt.show()

    return arima_forecast, stderr

# Example usage:
# Assuming you have a DataFrame `df` with a time series column `column`
# arima_forecast = arima_predict(df, 'column_name', 0, 100, 10, plot=True)