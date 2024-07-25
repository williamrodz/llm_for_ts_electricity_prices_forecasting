import pandas as pd
from wrappers.chronos_wrapper import chronos_predict
import matplotlib.pyplot as plt

def preprocess_series(series, window_size=3):
    # Ensure the length of the series is a multiple of window_size
    trimmed_length = len(series) - len(series) % window_size
    trimmed_series = series[:trimmed_length]
    
    # Reshape the series into a 2D array where each row is a window
    reshaped_series = trimmed_series.values.reshape(-1, window_size)
    
    # Compute the mean of each window
    averaged_series = reshaped_series.mean(axis=1)

    return pd.Series(averaged_series)



# Test the function
def test_on_passengers():
  passengers_data_table = df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")
  passengers_column = passengers_data_table["#Passengers"]
  length_passengers_data = len(passengers_column)
  context_range = (0, length_passengers_data * 3 // 4)
  prediction_length = 12
  chronos_predict(passengers_column, context_range[-1], context_range, prediction_length,autoregressions=2)

  # Test on Mackey-Glass time series
def test_on_mackey_glass():
  mackey_ts = pd.read_csv("data/mackey_glass_time_series.csv")
  # processed_series = preprocess_series(mackey_ts['Value'][3000:6000])

  # Graph the time series
  plt.figure(figsize=(10, 5))
  plt.plot(range(3000),mackey_ts['Value'][3000:6000])
  plt.title('Mackey-Glass Time Series')
  plt.xlabel('Time')
  plt.ylabel('x(t)')
  plt.show()
  # chronos_predict(processed_series, (30,120), 64,5)
  #chronos_predict(processed_series, (30,120), 64,10)



if __name__ == "__main__":
  print(
    """
┏┳┓┏┓  ┏┓               
 ┃ ┗┓  ┣ ┏┓┏┓┏┓┏┏┓┏╋┏┓┏┓
 ┻ ┗┛  ┻ ┗┛┛ ┗ ┗┗┻┛┗┗ ┛ 
                        """
  )
  test_on_mackey_glass()




