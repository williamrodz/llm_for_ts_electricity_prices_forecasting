import pandas as pd
import numpy as np

def calculate_mse(actual_values, predicted_values):
  cum_sum = 0
  n = len(predicted_values)
  for (actual,expected) in zip(actual_values,predicted_values):
    error = (expected - actual) ** 2
    cum_sum += error
  return cum_sum / n

def calculate_rmse(actual_values, predicted_values):
  return sqrt(calculate_mse(actual_values, predicted_values))

def mean_absolute_percentage_error(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def find_first_occurrence_index(df, date_string, date_column):
    """
    Finds the index of the first occurrence of a specified date in a given column of a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to search in.
    date_string (str): The date string to find (e.g., "2020-01-02").
    date_column (str): The column name where the date should be searched.

    Returns:
    int: The index of the first occurrence of the specified date, or -1 if not found.
    """
    # Convert the date string to a pandas Timestamp
    target_date = pd.to_datetime(date_string)

    # Find the first occurrence of the target date in the specified column
    mask = df[date_column] == target_date

    # Get the index of the first occurrence
    if mask.any():
        return df.index[mask][0]
    else:
        return -1
        

  