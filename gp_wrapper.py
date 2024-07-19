from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, ConstantKernel as C
import utils 
import numpy as np
import matplotlib.pyplot as plt
from constants import *

def gp_predict(df, column, date_column, training_start_index, training_end_index, prediction_length,plot=True):
  # Check for string date
  if isinstance(training_start_index, str) or isinstance(training_end_index, str):
    raise ValueError("Both training_start_index and training_end_index must be integers")

  # Extract the data
  training_df = df[training_start_index:training_end_index]
  length_scale = np.mean(training_df[column].values) * 1

  # Define the kernel
  kernel = RBF()

  # Create Gaussian Process Regressor
  gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

  train_data_time_steps_array = np.array(range(training_start_index,training_end_index)).reshape(-1, 1)
  training_y_values_array = np.array(training_df[column])
  # Fit to the training data
  gp.fit(train_data_time_steps_array, training_y_values_array)

  # Make predictions
  forecast_index = np.array(range(train_data_time_steps_array[-1][0], train_data_time_steps_array[-1][0] + prediction_length))
  y_pred, sigma = gp.predict(forecast_index.reshape(-1, 1), return_std=True)

  actual_values = df[forecast_index[0]:forecast_index[-1]][column]

  # Plot the original data and the forecast
  if plot:
    n = len(df)
    left_most_index = max(0, training_start_index - 2 * prediction_length)
    right_most_index = min(n, training_end_index + 2 * prediction_length)

    plt.figure(figsize=(8, 4))
    plt.title("GP Forecast")
    plt.plot(df[column][left_most_index: training_start_index], color="royalblue", label="Historical Data")
    plt.plot(df[column][training_start_index: training_end_index], color="green", label="Context")
    plt.plot(df[column][forecast_index[0]: right_most_index], color="royalblue", label="Post Context Historical Data")
    plt.plot(forecast_index, y_pred, color="tomato", label="GP Forecast")
    plt.legend()
    plt.show()

  return y_pred

