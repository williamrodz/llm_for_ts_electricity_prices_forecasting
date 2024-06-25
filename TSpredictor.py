import pandas as pd
import torch
import numpy as np
from chronos import ChronosPipeline
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import pmdarima as pm


pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cpu",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)
CHRONOS_NUM_SAMPLES_DEFAULT = 20

"""

input: 
    model:
      Identifies the model for time series prediction 
      e.g. Chronos, S-ARIMA, etc
    data: [float]
      Represents the time series data
    context_range: [int]
      Range of data points to use for prediction
    prediction_length: int
      Number of data points to predict
    autoregressions: int
      Number of autoregressions to create after the initial prediction window

output:
    [float]
      Returns the predicted values
    
    Not returned, but printed:
      graph of the predicted values against 80% prediction interval

"""
def predict(model:str,
            input_data:[float],
            context_range:[int],
            prediction_length,
            autoregressions:int=0):
  
  # Determine size of inputed time series
  print("Predicting time series")
  n = len(input_data)
  (context_start_index, context_end_index) = context_range
  (forecast_start_index, forecast_end_index) = (context_end_index, context_end_index+prediction_length)
  forecast_range = range(forecast_start_index, forecast_end_index)

  if n < 2:
    raise ValueError("Time series data has only one data point or is empty")
  elif not 0 <= context_start_index < context_end_index <= n:
    raise ValueError("Invalid context range")

  print(f"There are {n} data points in the time series.")
  print(f"Will be using {context_range} data points for predicting values at  t={forecast_range[0]} to t={forecast_range[-1]}")


  # Check if model is available
  if model == "Chronos":
    # Use Chronos model to predict time series
    print("Using Chronos model to predict time series")
    # Convert data to tensor
    context_slice = input_data[context_start_index:context_end_index]
    data_tensor = torch.tensor(context_slice)
    # Predict time series

    forecast = pipeline.predict(
        context=data_tensor,
        prediction_length=prediction_length,
        num_samples=CHRONOS_NUM_SAMPLES_DEFAULT
    )

    # Plot the predicted values and get intervals
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(input_data, color="royalblue", label="Input data")
    plt.plot(forecast_range, median, color="tomato", label="median forecast")
    plt.fill_between(forecast_range, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
    plt.legend()
    plt.grid()
    plt.show()
    return forecast[0].numpy()


    
  elif model == "S-ARIMA":
    # Use S-ARIMA model to predict time series
    print("Using S-ARIMA model to predict time series")
    # Convert data to pandas series
    data_series = pd.Series(data)
    # Fit the model
    model = pm.auto_arima(data_series, seasonal=False, stepwise=True, suppress_warnings=True)
    # Predict time series
    predictions = model.predict(n_periods=prediction_length)
    print(f"Predicted values: {predictions}")
  else:
    raise ValueError("Invalid model")



if __name__ == "__main__":
  print(
    """
┏┳┓┏┓  ┏┓               
 ┃ ┗┓  ┣ ┏┓┏┓┏┓┏┏┓┏╋┏┓┏┓
 ┻ ┗┛  ┻ ┗┛┛ ┗ ┗┗┻┛┗┗ ┛ 
                        """
  )

  # Test the function
  passengers_data_table = df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")
  passengers_column = passengers_data_table["#Passengers"]
  length_passengers_data = len(passengers_column)
  context_range = (0, length_passengers_data)
  prediction_length = 12
  predict("Chronos", passengers_column, context_range, prediction_length)