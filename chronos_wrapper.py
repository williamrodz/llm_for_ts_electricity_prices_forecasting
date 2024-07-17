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
      pipeline = ChronosPipeline.from_pretrained(
          f"amazon/chronos-t5-{version}",
          device_map=DEVICE_MAP,  
          torch_dtype=torch.bfloat16,
      )

    # print(" -------- PREDICT RUN ---------")
    # print("Parameters are:")
    # print(f"""
    # input_data: {len(input_data)}\n
    # context_range: {context_range}\n
    # prediction_length: {prediction_length}\n
    # autoregressions: {(autoregressions)}\n
    # median_predictions: {len(median_predictions)}\n
    # low_predictions: {len(low_predictions)}\n
    # high_predictions: {len(high_predictions)}\n
    # initial_context_start: {initial_context_start}\n
    # initial_context_end: {initial_context_end}\n

    # """)
    # print("----------")
    # print("")
    
    # Determine size of input time series
    n = len(input_data)

    if type(context_start_index) == str:
      # convert dates to index
      mask = (input_data[DATE_COLUMN] >= context_start_index) & (input_data[DATE_COLUMN] <= context_end_index)
      training_df = input_data.loc[mask]
      context_start_index = utils.find_first_occurrence_index(input_data, context_start_index,DATE_COLUMN)
      context_end_index = utils.find_first_occurrence_index(input_data, context_end_index,DATE_COLUMN)    

    context_slice = input_data[column][context_start_index:context_end_index]
    context_data_tensor = torch.tensor(context_slice.tolist())

    # Predict time series
    forecast = pipeline.predict(
        context=context_data_tensor,
        prediction_length=prediction_length,
        num_samples=CHRONOS_NUM_SAMPLES_DEFAULT
    )

    # Get the quantiles for the prediction interval
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    # Append predictions
    median_predictions = median
    low_predictions = low
    high_predictions = high  

    num_median_predictions = len(median_predictions)
    
    if plot:
      plt.figure(figsize=(8, 4))
      plt.title("Chronos Forecast")
      plt.plot(range(context_start_index), input_data[column][:context_start_index], color="royalblue", label="Reference data")
      plt.plot(range(context_start_index, context_end_index,), input_data[column][context_start_index:context_end_index], color="green", label="Context data")
      plt.plot(range(context_end_index,n), input_data[column][context_end_index:], color="royalblue", label="Reference data")
      plt.plot(range(context_end_index, context_end_index + num_median_predictions), median_predictions, color="tomato", label="Median forecast")
      plt.fill_between(range(context_end_index, context_end_index + num_median_predictions), low_predictions, high_predictions, color="tomato", alpha=0.3, label="80% prediction interval")
      plt.legend()
      plt.grid()
      plt.show()

    return median_predictions
