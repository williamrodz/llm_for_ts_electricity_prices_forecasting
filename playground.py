import pandas as pd
from LLMTS import chronos_predict
import matplotlib.pyplot as plt


# Test the function
def test_on_passengers():
  passengers_data_table = df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")
  passengers_column = passengers_data_table["#Passengers"]
  length_passengers_data = len(passengers_column)
  context_range = (0, length_passengers_data * 3 // 4)
  prediction_length = 12
  chronos_predict("Chronos", passengers_column, context_range[-1], context_range, prediction_length,autoregressions=2)

  # Test on Mackey-Glass time series
def test_on_mackey_glass():
  mackey_ts = pd.read_csv("mackey_glass_time_series.csv")
  # Graph the time series
  # plt.figure(figsize=(10, 5))
  # plt.plot(mackey_ts['Time'], mackey_ts['Value'])
  # plt.title('Mackey-Glass Time Series')
  # plt.xlabel('Time')
  # plt.ylabel('x(t)')
  # plt.show()
  chronos_predict(mackey_ts['Value'][5600:6100], (0,300), 64,3)

if __name__ == "__main__":
  print(
    """
┏┳┓┏┓  ┏┓               
 ┃ ┗┓  ┣ ┏┓┏┓┏┓┏┏┓┏╋┏┓┏┓
 ┻ ┗┛  ┻ ┗┛┛ ┗ ┗┗┻┛┗┗ ┛ 
                        """
  )
  test_on_mackey_glass()




