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

  # Define the kernel
  kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2)) + Matern(length_scale=1.0, nu=1.5) + ExpSineSquared(length_scale=1.0, periodicity=3.0)

  # Create Gaussian Process Regressor
  gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

  x_train = np.array(range(len(training_df))).reshape(-1, 1)

  # Fit to the training data
  gp.fit(x_train, training_df[column])

  # Make predictions
  forecast_index = df.index[training_end_index:training_end_index + prediction_length]

  y_pred, sigma = gp.predict(forecast_index.to_numpy().reshape(-1, 1), return_std=True)

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
    plt.plot(forecast_index, y_pred, color="tomato", label="Median Forecast")
    plt.legend()
    plt.show()

  return y_pred

