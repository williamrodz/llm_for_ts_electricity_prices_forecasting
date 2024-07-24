import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline
from chronos_wrapper import chronos_predict
from sarima_wrapper import sarima_predict
from gp_wrapper import gp_predict
from lstm_wrapper import lstm_predict
from tqdm import tqdm
from constants import *
import json
import time
import os

# Custom JSON Encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Function to save dictionary as a JSON file
def save_dict_to_json(dictionary, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(dictionary, json_file, indent=4, cls=NumpyEncoder)

def calculate_mse(actual_values, predicted_values):
  cum_sum_of_errors = 0
  n = len(predicted_values)
  for (actual,expected) in zip(actual_values,predicted_values):
    error = (expected - actual) ** 2
    cum_sum_of_errors += error
  
  mse = cum_sum_of_errors / n
  return mse

def calculate_nmse(y_true, y_pred):
    """
    Calculate the Normalized Mean Square Error (NMSE).
    
    NMSE = MSE / var(y_true)
    
    :param y_true: array-like, true values
    :param y_pred: array-like, predicted values
    :return: float, normalized mean square error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mse = np.mean((y_true - y_pred) ** 2)
    variance_of_true_values = np.var(y_true)
    
    return mse / variance_of_true_values

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

def x(results):
  # Extract the values and keys
  keys = list(results.keys())
  values = list(results.values())

  # Normalize the values for color mapping
  norm = plt.Normalize(min(values), max(values))
  colors = plt.cm.RdYlGn_r(norm(values))

  # Create a horizontal bar plot
  plt.figure(figsize=(10, 6))
  bars = plt.barh(keys, values, color=colors)

  # Add color bar for reference
  sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
  sm.set_array([])
  plt.colorbar(sm, orientation='vertical', label='Error Value')

  # Add values at the end of the bars
  for bar in bars:
      plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
              f'{bar.get_width():.2f}', va='center')

  plt.xlabel('Error Value')
  plt.title('Error Values for Different Models')
  plt.gca().invert_yaxis()  # Highest value at the top
  plt.show()

def sliding_window_analysis(df,column,context_length,prediction_length):
  # |S---F---P---|
  cum_sum_mse_sarima = 0
  cum_sum_mse_chronos = 0
  cum_sum_mse_gp = 0
  
  cum_sum_nmse_sarima = 0
  cum_sum_nmse_chronos = 0
  cum_sum_nmse_gp = 0

  fail_count = 0
  # Loop with tqdm progress bar

  num_possible_iterations = len(df) - context_length - prediction_length + 1
  print("Starting sliding window analysis")
  print("Note: This may take a while especially with a smaller context window and longer dataset.")
  print(f"- Algorithm:                SARIMA, Chronos, GP")
  print(f"- Context length:           {context_length}")
  print(f"- Prediction length:        {prediction_length}")
  print(f"- # of Possible iterations: {num_possible_iterations}")
  print("")

  for i in tqdm(range(0, num_possible_iterations)):
      context_start = i
      context_finish = i + context_length
      try:
          results = compare_prediction_methods(df, column, context_start, context_finish, prediction_length, plot=False)
        
          cum_sum_mse_sarima += results['mse_sarima']
          cum_sum_mse_chronos += results['mse_chronos']
          cum_sum_mse_gp += results['mse_gp']

          cum_sum_nmse_sarima += results['nmse_sarima']
          cum_sum_nmse_chronos += results['nmse_chronos']
          cum_sum_nmse_gp += results['nmse_gp']
      except Exception as e:
          fail_count += 1
          print(f"Iteration {i} failed with error: {e}")
      
  print(f"Number of failed iterations: {fail_count}")
  
  results = {
    'cum_sum_mse_sarima':cum_sum_mse_sarima,
    'cum_sum_mse_chronos':cum_sum_mse_chronos,
    'cum_sum_mse_gp':cum_sum_mse_gp,
    'cum_sum_nmse_sarima':cum_sum_nmse_sarima,
    'cum_sum_nmse_chronos':cum_sum_nmse_chronos,
    'cum_sum_nmse_gp':cum_sum_nmse_gp
  }
  # Plot final errors
  plot_error_comparison(results)
  return results

def sliding_window_analysis_for_algorithm(algo, data_title, df,column,context_length, prediction_length,plot=False):
  num_possible_iterations = len(df) - context_length - prediction_length + 1
  cum_mse = 0
  ledger_mse = np.array([])
  cum_nmse = 0
  ledger_nmse = np.array([])

  if algo.startswith("chronos_"):
    # Extract the part after "chronos_"
    suffix = algo[len("chronos_"):]
    
    # Define the valid sizes
    valid_chronos_sizes = {"tiny", "mini", "small", "base", "large"}
    
    if suffix in valid_chronos_sizes:
      #initialize chronos pipeline
      print("= = = = >")
      print(f"Initializing Chronos {suffix} pipeline...\n")
      pipeline = ChronosPipeline.from_pretrained(
      f"amazon/chronos-t5-{suffix}",
      device_map=DEVICE_MAP,  
      torch_dtype=torch.bfloat16,
      )
    else:
      raise ValueError(f"Invalid Chronos model size: {suffix}. Valid sizes are: {valid_chronos_sizes}")
  welcome_message = "- - - - - - - - -- - - - - - - - - - - - - - - \n"
  welcome_message += f"Starting sliding window analysis for {algo}\n"
  welcome_message += ("Note: This will take more time with a smaller context window and longer dataset.\n")
  welcome_message += (f"- Algorithm:                {algo}\n")
  welcome_message += (f"- Context length:           {context_length}\n")
  welcome_message += (f"- Prediction length:        {prediction_length}\n")
  welcome_message += (f"- Data Title:               {data_title}\n")
  welcome_message += (f"- Dataset length:           {len(df)}\n")
  welcome_message += (f"- # of possible iterations: {num_possible_iterations}\n")
  print(welcome_message)

  num_successful_runs = 0 
  start_time = time.time()

  for i in tqdm(range(0, num_possible_iterations)):
      context_start = i
      context_finish = i + context_length

      # Obtain predictions
      algo_predictions = None
 
      if algo.startswith("chronos_"):
        algo_predictions = chronos_predict(
          df,
          column,
          context_start,context_finish,
          prediction_length,
          plot=plot,
          pipeline=pipeline
          )

      elif algo == "sarima":
        algo_predictions = sarima_predict(df,column,context_start, context_finish,prediction_length,plot=plot)
      elif algo == "gp":
          algo_predictions = gp_predict(df,column, context_start, context_finish, prediction_length, plot=plot)
      else:
        raise ValueError(f"Invalid algorithm {algo}")
      
      # Obtain Actual Values
      forecast_start_index = context_finish
      forecast_end_index = forecast_start_index + prediction_length
      n_predictions = len(algo_predictions)

      actual_values = df[forecast_start_index:forecast_end_index][column]
      n_actual_values = len(actual_values)

      if not n_actual_values == n_predictions:
        raise ValueError(f"Unequal lengths {n_actual_values} != {n_predictions}")
      
      mse = calculate_mse(actual_values, algo_predictions)
      if not np.isnan(mse):
        cum_mse += mse
      np.append(ledger_mse,mse)

      nmse = calculate_nmse(actual_values, algo_predictions)
      if not np.isnan(nmse):
        cum_nmse += nmse
      np.append(ledger_nmse,nmse)

      if not np.isnan(mse) and not np.isnan(nmse):
        num_successful_runs += 1

  end_time = time.time()

  # Calculate elapsed time in seconds
  elapsed_seconds = end_time - start_time

  # Convert elapsed time to hours
  elapsed_hours = elapsed_seconds / 3600

  # Mean MSE and NMSE
  mean_mse = np.mean(ledger_mse[~np.isnan(ledger_mse)])
  mean_nmse = np.mean(ledger_nmse[~np.isnan(ledger_nmse)])

  # Save results in a txt file
  algo_results = {
    "algorithm":algo,
    "context_length":context_length,
    "prediction_length":prediction_length,
    "data_title":data_title,
    "dataset_length":len(df),
    "elapsed_hours":elapsed_hours,
    "cum_mse":cum_mse,
    "cum_nmse":cum_nmse,
    "mean_mse":mean_mse,
    "mean_nmse":mean_nmse,
    "num_possible_iterations":num_possible_iterations,
    "num_successful_runs":num_successful_runs,
    "successful_run_percentage": num_successful_runs / num_possible_iterations * 100,
    "ledger_mse":ledger_mse,
    "ledger_nmse":ledger_nmse,    
    }
  output_message = f"\nResults for {algo}:\n"

  for key,value in algo_results.items():
    if key == "ledger_mse" or key == "ledger_nmse":
      continue
    output_message += (f"- {key}: {value}\n")
  print(output_message)
  
  # Save results to a text file
  timestamp = pd.Timestamp.now()
  # Create the directory if it doesn't exist
  os.makedirs(f"{RESULTS_FOLDER_NAME}/{algo}", exist_ok=True)  
  file_name = f"{RESULTS_FOLDER_NAME}/{algo}/{algo}|context_length|{context_length}|data_title|{data_title}|{timestamp}.txt"
  save_dict_to_json(algo_results, file_name)

  return algo_results

def compare_prediction_methods(df, data_column, date_column, context_start, context_finish, prediction_length, plot=True,methods=["chronos_mini","gp","sarima","lstm"]):
  # convert dates to an integer index
  if type(context_start) == str:
    context_start = find_first_occurrence_index(df, context_start, date_column)
    context_finish = find_first_occurrence_index(df, context_finish, date_column)

  # Determine Forecast Indices
  forecast_start_index = context_finish
  forecast_end_index = forecast_start_index + prediction_length
  
  joint_results = {}
  for method in methods:
    if method.startswith("chronos_"):
      # Extract the part after "chronos_"
      chronos_version = method[len("chronos_"):]
      
      # Define the valid sizes
      valid_chronos_sizes = {"tiny", "mini", "small", "base", "large"}
      
      if chronos_version in valid_chronos_sizes:
        chronos_predictions = chronos_predict(
          df,
          data_column,
          context_start,context_finish,
          prediction_length,
          plot=plot,
          version=chronos_version
          )
        joint_results[method] = {"predictions":chronos_predictions}
      else:
        raise ValueError(f"Invalid Chronos model size: {chronos_version}. Valid sizes are: {valid_chronos_sizes}")
    elif method == "sarima":
      sarima_predictions = sarima_predict(
        df,
        data_column,
        context_start,
        context_finish,
        prediction_length,
        plot=plot
        )
      joint_results[method] = {"predictions":sarima_predictions}
    elif method == "gp":
      gp_predictions = gp_predict(df, data_column, date_column, context_start, context_finish, prediction_length, plot=plot)
      joint_results[method] = {"predictions":gp_predictions}
    elif method == "lstm":
      lstm_predictions = lstm_predict(df, data_column, context_start, context_finish, prediction_length, plot=plot)
      joint_results[method] = {"predictions":lstm_predictions}

    actual_values = df[forecast_start_index:forecast_end_index][data_column]

    if not len(actual_values) == len(joint_results[method]["predictions"]):
      raise ValueError("Unequal lengths of comparison values and predictions")
  
    mse = calculate_mse(actual_values, joint_results[method]["predictions"])
    nmse = calculate_nmse(actual_values, joint_results[method]["predictions"])
    joint_results[method]["mse"] = mse
    joint_results[method]["nmse"] = nmse

  if plot:
    output_message = f"\nResults comparison for {data_column}:\n\n"
    output_message += ("MSE\n")
    for method in methods:
      output_message += f"- {method} MSE: {joint_results[method]['mse']}\n"
    output_message += ("NMSE\n")
    for method in methods:
      output_message += f"- {method} NMSE: {joint_results[method]['nmse']}\n"      
    print(output_message)
  # Plot the results in a bar chart    

  return joint_results    


def get_sub_df_from_index(df, date_column, start_index, end_index):
  output_df = None
  if type(start_index) == str:
    # convert dates to index
    mask = (df[date_column] >= start_index) & (df[date_column] <= end_index)
    output_df = df.loc[mask]
    start_index = find_first_occurrence_index(df, start_index,date_column)
    end_index = find_first_occurrence_index(df, end_index,date_column)
    # forecast_index = range(end_index + 1,end_index + 1 + prediction_length)
  elif not start_index is None and not end_index is None:
    output_df = df[start_index:end_index]
    # forecast_index = df.index[end_index:end_index + prediction_length]
  else:
    raise ValueError(f'start_index is {type(start_index)}. Must be int or str.')

  return output_df