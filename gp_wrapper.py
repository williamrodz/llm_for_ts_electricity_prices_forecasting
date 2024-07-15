from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, ConstantKernel as C
import utils 
import numpy as np
import matplotlib.pyplot as plt
from constants import *

def gp_predict(df, column, training_start_index, training_end_index, prediction_length,plot=True):
  # Check for string date
  if isinstance(training_start_index, str) and isinstance(training_end_index, str):
    training_start_index = utils.find_first_occurrence_index(df, training_start_index,DATE_COLUMN)
    training_end_index = utils.find_first_occurrence_index(df, training_end_index,DATE_COLUMN)
  elif isinstance(training_start_index, str) or isinstance(training_end_index, str):
    raise ValueError("Both training_start_index and training_end_index must be strings or integers")


  # Extract the data
  training_df = utils.get_sub_df_from_index(df, training_start_index, training_end_index)

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

  # print("y_pred")
  # print(y_pred)

  actual_values = df[forecast_index[0]:forecast_index[-1]][column]

  gp_mse = utils.calculate_mse(actual_values, y_pred)
  # print(f"GP MSE: {gp_mse}")

  # Plot the original data and the forecast
  if plot:
    plt.figure(figsize=(8, 4))
    plt.title("GP Forecast")
    plt.plot(df[column][: training_start_index], color="royalblue", label="historical data")
    plt.plot(df[column][training_start_index: training_end_index], color="green", label="historical data")
    plt.plot(forecast_index, y_pred, color="tomato", label="median forecast")
    plt.plot(df[column][forecast_index[0]:], color="royalblue", label="historical data")
    plt.legend()
    plt.show()

  return y_pred

