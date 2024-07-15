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


ON_MAC = True
MPS_COMPATIBLE = False
DEVICE_MAP = "cpu" # cuda, mps, cpu


# use "cpu" for CPU inference and "mps" for Apple Silicon

def chronos_predict(
  input_data: [float],
  column: str,
  context_range: [int],
  prediction_length: int,
  autoregressions: int = 0,
  median_predictions: [float] = pd.Series([]),
  low_predictions: [float] = pd.Series([]),
  high_predictions: [float] = pd.Series([]),
  initial_context_start: int = None,
  initial_context_end: int = None,
  plot=True
  ):
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
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
    (context_start_index, context_end_index) = context_range

    # print("Data")
    # print(input_data)
    if type(context_start_index) == str:
      # convert dates to index
      mask = (input_data[DATE_COLUMN] >= context_start_index) & (input_data[DATE_COLUMN] <= context_end_index)
      training_df = input_data.loc[mask]
      context_start_index = utils.find_first_occurrence_index(input_data, context_start_index,DATE_COLUMN)
      context_end_index = utils.find_first_occurrence_index(input_data, context_end_index,DATE_COLUMN)    

    if n < 2:
        raise ValueError("Time series data has only one data point or is empty")
    elif not 0 <= context_start_index < context_end_index <= (n + len(median_predictions)):
        raise ValueError("Invalid context range")

    # Convert data to tensor
    # print ("context_slice", type(context_slice), context_slice)

    extended_slice_by_predictions = pd.concat([input_data[column], median_predictions])
    context_slice = extended_slice_by_predictions[context_start_index:context_end_index]

    context_data_tensor = torch.tensor(context_slice.tolist(), dtype=torch.float32)

    # Predict time series
    forecast = pipeline.predict(
        context=context_data_tensor,
        prediction_length=prediction_length,
        num_samples=CHRONOS_NUM_SAMPLES_DEFAULT
    )

    # Get the quantiles for the prediction interval
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    # Append predictions
    median_predictions = pd.concat([median_predictions,pd.Series(median)])
    low_predictions = pd.concat([low_predictions,pd.Series(low)])
    high_predictions = pd.concat([high_predictions,pd.Series(high)])   

    # context_start_index in two lines below will need to be amended for recursion
    initial_context_start_to_carry = context_start_index if initial_context_start is None else initial_context_start
    initial_context_end_to_carry = context_end_index if initial_context_end is None else initial_context_end

    if autoregressions == 0:
        # graph
        # Plot the predicted values and prediction intervals

        num_median_predictions = len(median_predictions)
        
        # print("input_data[column][:initial_context_start_to_carry],")
        # print(input_data[column][:initial_context_start_to_carry],)
        if plot:
          plt.figure(figsize=(8, 4))
          plt.title("Chronos Forecast")
          plt.plot(range(initial_context_start_to_carry), input_data[column][:initial_context_start_to_carry], color="royalblue", label="Reference data")
          plt.plot(range(initial_context_start_to_carry, initial_context_end_to_carry,), input_data[column][initial_context_start_to_carry:initial_context_end_to_carry], color="green", label="Context data")
          plt.plot(range(initial_context_end_to_carry,n), input_data[column][initial_context_end_to_carry:], color="royalblue", label="Reference data")
          plt.plot(range(initial_context_end_to_carry, initial_context_end_to_carry + num_median_predictions), median_predictions, color="tomato", label="Median forecast")
          plt.fill_between(range(initial_context_end_to_carry, initial_context_end_to_carry + num_median_predictions), low_predictions, high_predictions, color="tomato", alpha=0.3, label="80% prediction interval")
          plt.legend()
          plt.grid()
          plt.show()

        # print("CONVERGED: Reached base case")
        return median_predictions

    elif autoregressions >= 1:
        # combine new prediction into input data AT the context's end
        new_context_range = (context_start_index + prediction_length, context_end_index + prediction_length)
        # recurse
        print("RECURSING: Autoregressions left:", autoregressions - 1)
        return chronos_predict(
          input_data,
          column,
          new_context_range,
          prediction_length,
          autoregressions - 1,
          median_predictions,
          low_predictions,
          high_predictions,
          initial_context_start_to_carry,
          initial_context_end_to_carry)

# def chronos_predict(input_data: [float],
#             context_range: [int],
#             prediction_length: int,
#             autoregressions: int = 0,
#             median_predictions: [float] = pd.Series([]),
#             low_predictions: [float] = pd.Series([]),
#             high_predictions: [float] = pd.Series([]),
#             initial_context_end: int = None
# ):