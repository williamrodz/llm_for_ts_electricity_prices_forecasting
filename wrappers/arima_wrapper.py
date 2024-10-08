import matplotlib.pyplot as plt
import pmdarima as pm
import statsmodels.api as sm
import utils
import matplotlib.dates as mdates

def arima_predict(df, column, training_start_index, training_end_index, prediction_length, plot=True, run_name=None):
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
    # Define indices for plotting
        n = len(df)
        left_most_index = max(0, training_start_index - 2 * prediction_length)
        right_most_index = min(n, training_end_index + 2 * prediction_length)
        
        # Map indices to dates
        reference_dates = [utils.map_timestep_to_date(idx) for idx in range(left_most_index, training_start_index)]
        context_dates = [utils.map_timestep_to_date(idx) for idx in range(training_start_index, training_end_index)]
        future_dates = [utils.map_timestep_to_date(idx) for idx in range(training_end_index, right_most_index)]
        prediction_dates = [utils.map_timestep_to_date(idx) for idx in range(training_end_index, training_end_index + prediction_length)]
        
        # Combine all dates together to calculate tick positions
        all_dates = reference_dates + context_dates + future_dates

        # Calculate positions for 0%, 25%, 50%, 75%, and 100% of the date range
        num_dates = len(all_dates)
        ticks_positions = [
            all_dates[0],                           # 0%
            all_dates[num_dates // 4],              # 25%
            all_dates[num_dates // 2],              # 50%
            all_dates[3 * num_dates // 4],          # 75%
            all_dates[-1]                           # 100%
        ]
        
        # Plotting with dates on the x-axis
        plt.figure(figsize=(10, 5))
        plt.plot(reference_dates, df[column][left_most_index:training_start_index], color="royalblue", label="Historical Data")
        plt.plot(context_dates, df[column][training_start_index:training_end_index], color="green", label="Context Data")
        plt.plot(future_dates, df[column][training_end_index:right_most_index], color="royalblue")
        plt.plot(prediction_dates, arima_forecast, color="tomato", label="ARIMA Median Forecast")
        plt.fill_between(prediction_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='tomato', alpha=0.2, label="95% Confidence Interval")
        
        # Add legend, labels, and grid
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Price per kWh (pence)")
        plt.grid()

        # Set custom x-ticks at the calculated positions (5 ticks)
        plt.gca().set_xticks(ticks_positions)

        # Format the x-axis with date labels
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        if run_name and type(run_name) == str and run_name != "":
            plt.savefig(f"results/plots/arima_{run_name}.png", dpi=300)

    return arima_forecast, stderr

# Example usage:
# Assuming you have a DataFrame `df` with a time series column `column`
# arima_forecast = arima_predict(df, 'column_name', 0, 100, 10, plot=True)