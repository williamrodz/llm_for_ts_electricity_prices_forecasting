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
import numpy as np
import torch
import matplotlib.pyplot as plt

def predict(model: str,
            input_data: [float],
            initial_context_end: int,
            context_range: [int],
            prediction_length: int,
            autoregressions: int = 0,
            median_predictions: [float] = pd.Series([]),
            low_predictions: [float] = pd.Series([]),
            high_predictions: [float] = pd.Series([])):
    print(" -------- PREDICT RUN ---------")
    print("Parameters are:")
    print("----------")
    print(f"model: {model}\ninput_data: {len(input_data)}\ninitial_context_end: {initial_context_end}\ncontext_range: {context_range}\nprediction_length: {prediction_length}\nautoregressions: {autoregressions}\nmedian_predictions: {median_predictions}\nlow_predictions: {low_predictions}\nhigh_predictions: {high_predictions}")
    print("----------")
    print("")
    
    # Determine size of input time series
    print("Predicting time series")
    n = len(input_data)
    (context_start_index, context_end_index) = context_range

    if n < 2:
        raise ValueError("Time series data has only one data point or is empty")
    elif not 0 <= context_start_index < context_end_index <= (n + len(median_predictions)):
        raise ValueError("Invalid context range")

    # Convert data to tensor
    context_slice = input_data[context_start_index:context_end_index]
    # print ("context_slice", type(context_slice), context_slice)
    print(median_predictions)

    extended_slice_by_predictions = pd.concat([context_slice, median_predictions])
    context_data_tensor = torch.tensor(extended_slice_by_predictions.tolist(), dtype=torch.float32)

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

    if autoregressions == 0:
        # graph
        # Plot the predicted values and prediction intervals

        num_median_predictions = len(median_predictions)
        

        plt.figure(figsize=(8, 4))
        plt.plot(range(n), input_data[:n], color="royalblue", label="Input data")
        plt.plot(range(initial_context_end, initial_context_end + num_median_predictions), median_predictions, color="tomato", label="Median forecast")
        plt.fill_between(range(initial_context_end, initial_context_end + num_median_predictions), low_predictions, high_predictions, color="tomato", alpha=0.3, label="80% prediction interval")
        plt.legend()
        plt.grid()
        plt.show()

        print("CONVERGED: Reached base case")
        return median_predictions

    elif autoregressions >= 1:
        # combine new prediction into input data AT the context's end
        new_context_range = (context_start_index + prediction_length, context_end_index + prediction_length)
        # recurse
        print("RECURSING: Autoregressions left:", autoregressions - 1)
        return predict(model, input_data, initial_context_end, new_context_range, prediction_length, autoregressions - 1, median_predictions, low_predictions, high_predictions)





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
  context_range = (0, length_passengers_data * 3 // 4)
  prediction_length = 12
  predict("Chronos", passengers_column, context_range[-1], context_range, prediction_length,autoregressions=2)