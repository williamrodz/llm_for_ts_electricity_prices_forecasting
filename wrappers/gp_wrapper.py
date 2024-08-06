from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, ConstantKernel as C, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt
from constants import *

def gp_predict(df, column, training_start_index, training_end_index, prediction_length, plot=True):
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

        plt.figure(figsize=(10, 5))
        plt.title("Gaussian Process Forecast")
        plt.plot(df[column][left_most_index: training_start_index], color="royalblue", label="Historical Data")
        plt.plot(df[column][training_start_index: training_end_index], color="green", label="Context")
        plt.plot(df[column][forecast_index[0]: right_most_index], color="royalblue", label="Post Context Historical Data")
        plt.plot(forecast_index, y_pred_rescaled, color="tomato", label="GP Forecast")
        plt.fill_between(forecast_index, y_pred_rescaled - 1.96 * sigma_rescaled, y_pred_rescaled + 1.96 * sigma_rescaled, color='tomato', alpha=0.2, label="95% Confidence Interval")
        plt.legend()
        plt.show()

    return y_pred_rescaled, sigma_rescaled
