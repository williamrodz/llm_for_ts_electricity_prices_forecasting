import torch
import numpy as np
from chronos import ChronosPipeline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc # used for increasing font size in plots

from constants import *
import utils

def chronos_predict(input_data: [float], column: str,
  context_start_index: int,
  context_end_index: int,
  prediction_length: int,
  pipeline=None,
  plot=True,
  version="small",
  run_name=None
  ):
    """
    input:
        input_data: [float]
            Represents the time series data
        column: str
            Represents the column in the DataFrame to predict
        context_start_index: int
            Represents the start index of the context data
        context_end_index: int
            Represents the end index of the context data
        prediction_length: int
            Number of data points to predict
        pipeline: ChronosPipeline
            Represents the Chronos pipeline to use
        plot: bool
            Determines whether to plot the forecast
        version: str
            Represents the version of the Chronos model to use
    output:
        median_predictions: [float]
            Returns the predicted median values
        std_devs: [float]
            Returns the standard deviations of the forecast

    """
    
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

    # Get the quantiles for the 80% prediction interval
    #low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    # Get the quantiles for the 95% prediction interval
    low, median, high = np.quantile(forecast[0].numpy(), [0.025, 0.5, 0.975], axis=0)    

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
      rc('font', size=15)
    
      # Map time steps to corresponding dates
      reference_dates = [utils.map_timestep_to_date(idx) for idx in range(left_most_index, context_start_index)]
      context_dates = [utils.map_timestep_to_date(idx) for idx in range(context_start_index, context_end_index)]
      future_dates = [utils.map_timestep_to_date(idx) for idx in range(context_end_index, right_most_index)]
      prediction_dates = [utils.map_timestep_to_date(idx) for idx in range(context_end_index, context_end_index + num_median_predictions)]
      
      # Combine all dates together to calculate tick positions
      all_dates = reference_dates + context_dates + future_dates
      
      # Calculate positions for 0%, 25%, 50%, 75%, and 100% of the date range
      num_dates = len(all_dates)
      ticks_positions = [
          all_dates[0],                           # 0%
          all_dates[num_dates // 4],              # 25%
          all_dates[num_dates // 2],              # 50%
          all_dates[3 * num_dates // 4],          # 75%
          all_dates[-1]                           # 100%
      ]
    
      # Plotting with dates on the x-axis
      plt.plot(reference_dates, input_data[column][left_most_index:context_start_index], color="royalblue", label="Historical Data")
      plt.plot(context_dates, input_data[column][context_start_index:context_end_index], color="green", label="Context Data")
      plt.plot(future_dates, input_data[column][context_end_index:right_most_index], color="royalblue")
      plt.plot(prediction_dates, median_predictions, color="tomato", label="Chronos Median Forecast")
      plt.fill_between(prediction_dates, low_predictions, high_predictions, color="tomato", alpha=0.3, label="95% Prediction Interval")
      plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

      # Add legend, labels, and grid
      plt.legend()
      plt.xlabel("Date")
      plt.ylabel("Price per kWh (pence)")
      plt.grid()

      # Set custom x-ticks at the calculated positions (5 ticks)
      plt.gca().set_xticks(ticks_positions)

      # Format the x-axis with date labels
      plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))


      # Rotate date labels for better readability
      plt.xticks(rotation=0)
      if run_name and type(run_name) == str and run_name != "":
        plt.savefig(f"results/plots/{version}_{run_name}.png", dpi=300)      
     # plt.show()

    return median_predictions, std_devs
