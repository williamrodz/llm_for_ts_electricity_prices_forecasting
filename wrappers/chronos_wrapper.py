import pandas as pd
import torch
import numpy as np
from chronos import ChronosPipeline
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import pmdarima as pm
import utils
from constants import *

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
import numpy as np
import torch
import matplotlib.pyplot as plt
from constants import *

def chronos_predict(
  input_data: [float],
  column: str,
  context_start_index: int,
  context_end_index: int,
  prediction_length: int,
  pipeline=None,
  plot=True,
  version="small"
  ):
    # Initialize pipeline if not provided
    if pipeline is None:
      print(f"= = = > Chronos pipeline not initialized. Firing up {version} pipeline. May take time..")
      print(f"Version is {version}")
      if "-" in version:
        # This is a custom model
        pipeline = ChronosPipeline.from_pretrained(
          f"froyoresearcher/{version}",
          device_map=DEVICE_MAP,  
          torch_dtype=torch.bfloat16,
        )
      else:
        # Cut out the chronos leading text
        chronos_version = version.split("_")[-1]
        pipeline = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{chronos_version}",
            device_map=DEVICE_MAP,  
            torch_dtype=torch.bfloat16,
        )

    # Determine size of input time series
    n = len(input_data)

    if type(context_start_index) == str or type(context_end_index) == str:
      raise ValueError("Both context_start_index and context_end_index must be integers")
      
    context_slice = input_data[column][context_start_index:context_end_index]
    context_data_tensor = torch.tensor(context_slice.tolist())

    # Predict time series
    forecast = pipeline.predict(
        context=context_data_tensor,
        prediction_length=prediction_length,
        num_samples=CHRONOS_NUM_SAMPLES_DEFAULT,
        limit_prediction_length=False
    )

    # Get the quantiles for the prediction interval
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    # Append predictions
    median_predictions = median
    low_predictions = low
    high_predictions = high

    # Compute the standard deviation
    # Calculate the standard deviation across the sample dimension
    std_devs = np.std(forecast[0].numpy(), axis=0)      

    num_median_predictions = len(median_predictions)

    # plot a window of actual data two prediction windows to each side
    left_most_index = max(0, context_start_index - 2 * prediction_length)
    right_most_index = min(n, context_end_index + 2 * prediction_length)
    
    if plot:
      print(version)
      plt.figure(figsize=(8, 4))
      plt.title("Chronos Forecast")
      plt.plot(range(left_most_index,context_start_index), input_data[column][left_most_index:context_start_index], color="royalblue", label="Reference data")
      plt.plot(range(context_start_index, context_end_index,), input_data[column][context_start_index:context_end_index], color="green", label="Context data")
      plt.plot(range(context_end_index,right_most_index), input_data[column][context_end_index:right_most_index], color="royalblue", label="Post-Context Reference data")
      plt.plot(range(context_end_index, context_end_index + num_median_predictions), median_predictions, color="tomato", label="Chronos Median forecast")
      plt.fill_between(range(context_end_index, context_end_index + num_median_predictions), low_predictions, high_predictions, color="tomato", alpha=0.3, label="80% Prediction Interval")
      plt.legend()
      plt.grid()
      plt.show()

    return median_predictions, std_devs
