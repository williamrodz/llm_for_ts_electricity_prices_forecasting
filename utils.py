import pandas as pd
import numpy as np
from chronos_wrapper import chronos_predict
from sarima_wrapper import sarima_predict

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


def compare_chronos_to_sarima(df,column,context_start,context_finish, prediction_length):
  sarima_predictions = sarima_predict(
    df,
    'Daily average',
    context_start,
    context_finish,
    prediction_length
    )
  chronos_predictions = chronos_predict(
            df,
            'Daily average',
            [context_start,context_finish],
            prediction_length
            )

  # Forecast

  forecast_start_index = find_first_occurrence_index(df, context_finish, "Date") + 1
  forecast_end_index = forecast_start_index + prediction_length

  actual_values = df[forecast_start_index:forecast_end_index][column]

  if not len(actual_values) == len(sarima_predictions) == len(chronos_predictions):
    print("Unequal lengths of comparison values and predictions")
  
  sarima_mse = calculate_mse(actual_values, sarima_predictions)
  chronos_mse = calculate_mse(actual_values, chronos_predictions)
  print("--- RESULTS --- ")
  print (f"sarima_mse {sarima_mse}")
  print (f"chronos_mse {chronos_mse}")


def get_sub_df_from_index(df, start_index, end_index):
  output_df = None
  if type(start_index) == str:
    # convert dates to index
    mask = (df[DATE_COLUMN] >= start_index) & (df[DATE_COLUMN] <= end_index)
    output_df = df.loc[mask]
    start_index = find_first_occurrence_index(df, start_index,DATE_COLUMN)
    end_index = find_first_occurrence_index(df, end_index,DATE_COLUMN)
    # forecast_index = range(end_index + 1,end_index + 1 + prediction_length)
  elif not start_index is None and not end_index is None:
    output_df = df[start_index:end_index]
    # forecast_index = df.index[end_index:end_index + prediction_length]
  else:
    raise ValueError(f'start_index is {type(start_index)}. Must be int or str.')

  return output_df