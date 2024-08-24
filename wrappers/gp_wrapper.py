from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, ConstantKernel as C, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt
from constants import *
import utils
import matplotlib.dates as mdates

def gp_predict(df, column, training_start_index, training_end_index, prediction_length, plot=True, run_name=None):
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
        y_pred_rescaled: [float]
            Returns the predicted values
        sigma_rescaled: [float]
            Returns the standard deviations of the forecast
    """
    # Check for string date
    if isinstance(training_start_index, str) or isinstance(training_end_index, str):
        raise ValueError("Both training_start_index and training_end_index must be integers")

    # Extract the data
    training_df = df[training_start_index:training_end_index]
    
    # Move the training data to zero mean
    mean = training_df[column].mean()
    std = training_df[column].std()

    if std == 0:
        raise ValueError("Standard deviation is zero, cannot normalize.")
    
    normalized_training_data = (training_df[column] - mean) / std

    # Define the kernel
    T = training_end_index - training_start_index + 1
    length_scale = T / 10
    signal_variance = std**2
    noise_level = std / 10

    if column == 'Daily average':
        periodicity = 365
    elif column == 'Price_Ex_VAT':
        periodicity = 48
    else:
        raise ValueError("Unrecognized column name, cannot provide periodicity")

    # Define the periodic kernel
    periodic_kernel = ExpSineSquared(length_scale=length_scale, periodicity=periodicity, length_scale_bounds=(1e-2, 1e2), periodicity_bounds=(1e-2, 1e2))

    kernel = C(signal_variance, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2)) * periodic_kernel + WhiteKernel(noise_level, (1e-10, 1e1))

    # Create Gaussian Process Regressor
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    train_data_time_steps_array = np.array(range(training_start_index, training_end_index)).reshape(-1, 1)

    # Fit to the training data
    gp.fit(train_data_time_steps_array, normalized_training_data)

    # Make predictions
    forecast_index = np.array(range(train_data_time_steps_array[-1][0], train_data_time_steps_array[-1][0] + prediction_length))
    y_pred, sigma = gp.predict(forecast_index.reshape(-1, 1), return_std=True)

    y_pred_rescaled = y_pred * std + mean
    sigma_rescaled = sigma * std  # Rescale the uncertainty

    # Plot the original data and the forecast
    if plot:
        n = len(df)
        left_most_index = max(0, training_start_index - 2 * prediction_length)
        right_most_index = min(n, training_end_index + 2 * prediction_length)

        # Map time steps to dates
        historical_dates = [utils.map_timestep_to_date(idx) for idx in range(left_most_index, training_start_index)]
        context_dates = [utils.map_timestep_to_date(idx) for idx in range(training_start_index, training_end_index)]
        forecast_dates = [utils.map_timestep_to_date(idx) for idx in range(forecast_index[0], right_most_index)]
        prediction_dates = [utils.map_timestep_to_date(idx) for idx in forecast_index]

        plt.figure(figsize=(10, 5))
        plt.plot(historical_dates, df[column][left_most_index: training_start_index], color="royalblue", label="Historical Data")
        plt.plot(context_dates, df[column][training_start_index: training_end_index], color="green", label="Context Data")
        plt.plot(forecast_dates, df[column][forecast_index[0]: right_most_index], color="royalblue")
        plt.plot(prediction_dates, y_pred_rescaled, color="tomato", label="GP (Composite Kernel) Forecast")
        plt.fill_between(prediction_dates, y_pred_rescaled - 1.96 * sigma_rescaled, y_pred_rescaled + 1.96 * sigma_rescaled, color='tomato', alpha=0.2, label="95% Confidence Interval")

        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Price per kWh (pence)")

        # Combine all dates together to calculate tick positions
        all_dates = historical_dates + context_dates + forecast_dates + prediction_dates

        # Calculate positions for 0%, 25%, 50%, 75%, and 100% of the date range
        num_dates = len(all_dates)
        ticks_positions = [
            all_dates[0],                           # 0%
            all_dates[num_dates // 4],              # 25%
            all_dates[num_dates // 2],              # 50%
            all_dates[3 * num_dates // 4],          # 75%
            all_dates[-1]                           # 100%
        ]
        # Set custom x-ticks at the calculated positions (5 ticks)
        plt.gca().set_xticks(ticks_positions)                

        # Format the x-axis with date labels
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        if run_name and type(run_name) == str and run_name != "":
            plt.savefig(f"results/plots/gp_{run_name}.png", dpi=300)
        # plt.show()   

    return y_pred_rescaled, sigma_rescaled
