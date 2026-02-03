import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline, Chronos2Pipeline
from wrappers.chronos_wrapper import chronos_predict
from wrappers.chronos_2_wrapper import chronos_2_predict
#from wrappers.sarima_wrapper import sarima_predict
from wrappers.gp_wrapper import gp_predict
from wrappers.lstm_wrapper import lstm_predict
from tqdm import tqdm
from constants import *
import json
import time
import os


def check_timestamp_gaps(df, timestamp_column='timestamp', expected_freq='5min'):
    """
    Check for gaps in timestamp data and report statistics.

    Args:
        df: DataFrame with a timestamp column
        timestamp_column: Name of the timestamp column
        expected_freq: Expected frequency between timestamps (e.g., '5min', '1h')

    Returns:
        Dictionary with gap statistics
    """
    df = df.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df.sort_values(timestamp_column)

    # Calculate time differences between consecutive rows
    time_diffs = df[timestamp_column].diff().dropna()

    # Convert expected frequency to timedelta
    expected_td = pd.Timedelta(expected_freq)

    # Find gaps (differences larger than expected)
    gaps = time_diffs[time_diffs > expected_td]

    # Calculate expected number of intervals
    total_time_span = df[timestamp_column].max() - df[timestamp_column].min()
    expected_td = pd.Timedelta(expected_freq)
    expected_intervals = int(total_time_span / expected_td) + 1
    filled_intervals = len(df)
    fill_percentage = (filled_intervals / expected_intervals) * 100 if expected_intervals > 0 else 0

    # Calculate statistics
    stats = {
        'total_rows': len(df),
        'expected_freq': expected_freq,
        'num_gaps': len(gaps),
        'total_time_span': total_time_span,
        'expected_intervals': expected_intervals,
        'filled_intervals': filled_intervals,
        'fill_percentage': fill_percentage,
        'min_diff': time_diffs.min(),
        'max_diff': time_diffs.max(),
        'mean_diff': time_diffs.mean(),
        'median_diff': time_diffs.median(),
    }

    print("=" * 60)
    print("Timestamp Gap Analysis")
    print("=" * 60)
    print(f"Total rows:          {stats['total_rows']}")
    print(f"Time span:           {stats['total_time_span']}")
    print(f"Expected frequency:  {expected_freq}")
    print(f"Min time diff:       {stats['min_diff']}")
    print(f"Max time diff:       {stats['max_diff']}")
    print(f"Mean time diff:      {stats['mean_diff']}")
    print(f"Median time diff:    {stats['median_diff']}")
    print(f"Number of gaps:      {stats['num_gaps']} (intervals > {expected_freq})")

    if len(gaps) > 0:
        print("\nTop 10 largest gaps:")
        print("-" * 40)
        gap_df = pd.DataFrame({
            'timestamp': df[timestamp_column].iloc[gaps.index],
            'gap_duration': gaps.values
        }).sort_values('gap_duration', ascending=False).head(10)
        for _, row in gap_df.iterrows():
            print(f"  {row['timestamp']}: gap of {row['gap_duration']}")

    # ASCII visualization of data density over time
    print("\nData density timeline (each char = 1 hour):")
    print("-" * 60)

    # Create hourly buckets
    start_time = df[timestamp_column].min().floor('h')
    end_time = df[timestamp_column].max().ceil('h')
    total_hours = int((end_time - start_time).total_seconds() / 3600)

    # Count data points per hour
    df_temp = df.copy()
    df_temp['hour_bucket'] = df_temp[timestamp_column].dt.floor('h')
    hourly_counts = df_temp.groupby('hour_bucket').size()

    # Determine display width (max 80 chars for timeline)
    display_width = min(total_hours, 120)
    hours_per_char = max(1, total_hours // display_width)

    # Build ASCII representation
    # Calculate max possible data points per bucket (12 five-minute slots per hour)
    slots_per_bucket = hours_per_char * 12

    timeline = []
    for i in range(0, total_hours, hours_per_char):
        bucket_start = start_time + pd.Timedelta(hours=i)
        bucket_end = bucket_start + pd.Timedelta(hours=hours_per_char)

        # Count data points in this display bucket
        count = 0
        for h in range(hours_per_char):
            hour = bucket_start + pd.Timedelta(hours=h)
            if hour in hourly_counts.index:
                count += hourly_counts[hour]

        # Calculate fill percentage for this bucket
        bucket_fill_pct = (count / slots_per_bucket) * 100 if slots_per_bucket > 0 else 0

        # Map fill percentage to digit (0-9)
        # ' '=0%, 1=1-9%, 2=10-19%, 3=20-29%, ... 9=80%+
        if count == 0:
            timeline.append(' ')
        elif bucket_fill_pct < 10:
            timeline.append('1')
        elif bucket_fill_pct < 20:
            timeline.append('2')
        elif bucket_fill_pct < 30:
            timeline.append('3')
        elif bucket_fill_pct < 40:
            timeline.append('4')
        elif bucket_fill_pct < 50:
            timeline.append('5')
        elif bucket_fill_pct < 60:
            timeline.append('6')
        elif bucket_fill_pct < 70:
            timeline.append('7')
        elif bucket_fill_pct < 80:
            timeline.append('8')
        else:
            timeline.append('9')

    # Print with day markers
    timeline_str = ''.join(timeline)
    chars_per_day = max(1, 24 // hours_per_char)

    print(f"Start: {start_time}")
    print(f"End:   {end_time}")
    print(f"Scale: {hours_per_char} hour(s) per character")
    print()
    print("Legend: ' '=0%  1=1-9%  2=10-19%  3=20-29% ... 9=80%+")
    print()

    # Print timeline in rows of 60 chars with day labels
    row_width = 60
    for i in range(0, len(timeline_str), row_width):
        row = timeline_str[i:i+row_width]
        day_offset = (i * hours_per_char) // 24
        print(f"Day {day_offset:2d}: [{row}]")

    print()
    print(f"Expected intervals:  {stats['expected_intervals']}")
    print(f"Filled intervals:    {stats['filled_intervals']}")
    print(f"Fill percentage:     {stats['fill_percentage']:.1f}%")
    print("=" * 60)

    return stats


def resample_to_regular_intervals(df, timestamp_column='timestamp', freq='5min', fill_gaps='interpolate'):
    """
    Resample a DataFrame with irregular timestamps to regular intervals.

    Uses the mean of values within each interval for numeric columns.
    Optionally fills gaps to ensure perfectly regular timestamps (required by Chronos-2).

    Args:
        df: DataFrame with a timestamp column
        timestamp_column: Name of the timestamp column
        freq: Frequency string for resampling (e.g., '5min', '1h', '30min')
        fill_gaps: Strategy for filling gaps in the resampled data:
            - 'interpolate': Linear interpolation between known values (recommended)
            - 'ffill': Forward-fill with last known value
            - 'bfill': Back-fill with next known value
            - None: Drop empty intervals (results in irregular timestamps)

    Returns:
        DataFrame resampled to regular intervals with reset index.
        If fill_gaps is set, all intervals will be filled (no gaps).

    Note:
        For use with Chronos-2, fill_gaps must be set to a non-None value
        since Chronos-2 requires perfectly regular timestamp intervals.
    """
    df = df.copy()

    # Ensure timestamp is datetime
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Set timestamp as index for resampling
    df = df.set_index(timestamp_column)

    # Select only numeric columns for resampling
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Resample and take mean of each interval
    df_resampled = df[numeric_cols].resample(freq).mean()

    # Fill gaps based on strategy
    if fill_gaps == 'interpolate':
        df_resampled = df_resampled.interpolate(method='linear')
        # Handle edge NaNs that interpolate can't fill
        df_resampled = df_resampled.bfill().ffill()
    elif fill_gaps == 'ffill':
        df_resampled = df_resampled.ffill().bfill()
    elif fill_gaps == 'bfill':
        df_resampled = df_resampled.bfill().ffill()
    elif fill_gaps is None:
        # Drop rows where all values are NaN (empty intervals)
        df_resampled = df_resampled.dropna(how='all')
    else:
        raise ValueError(f"Invalid fill_gaps value: {fill_gaps}. Use 'interpolate', 'ffill', 'bfill', or None.")

    # Reset index to make timestamp a column again
    df_resampled = df_resampled.reset_index()

    return df_resampled


def add_weekday_column(df, timestamp_column='timestamp'):
    """
    Add a 'weekday' column to a DataFrame indicating if the timestamp falls on a weekday.

    Args:
        df: DataFrame with a timestamp column
        timestamp_column: Name of the timestamp column

    Returns:
        Copy of DataFrame with a new 'weekday' column (True for Mon-Fri, False for Sat-Sun)
    """
    df = df.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    # dayofweek: Monday=0, Sunday=6. Weekdays are 0-4.
    df['weekday'] = df[timestamp_column].dt.dayofweek < 5
    return df


def map_timestep_to_date(row_index):
    """
    Maps an integer row index to the corresponding date in the 'Valid_From_UTC' column.
    
    Args:
        row_index (int): The integer representing the row index.
        file_path (str): Path to the CSV file containing the date column.
    
    Returns:
        pd.Timestamp or None: The corresponding date in the 'Valid_From_UTC' column, or None if the index is out of bounds or if an error occurs.
    """
    half_hourly_prices = pd.read_csv(f"{DATA_FOLDER}/agile_octopus_london.csv")

    try:

        # Ensure 'Valid_From_UTC' is in datetime format
        if 'Valid_From_UTC' in half_hourly_prices.columns:
            if not pd.api.types.is_datetime64_any_dtype(half_hourly_prices['Valid_From_UTC']):
                # Attempt to infer the date format; if known, specify it explicitly, e.g., format='%Y-%m-%d %H:%M:%S'
                half_hourly_prices['Valid_From_UTC'] = pd.to_datetime(half_hourly_prices['Valid_From_UTC'], format="%m/%d/%y %H:%M")

            # Check if the row_index is within bounds
            if row_index >= 0 and row_index < len(half_hourly_prices):
                date_value = half_hourly_prices.loc[row_index, 'Valid_From_UTC']
                return date_value
            else:
                print(f"Row index {row_index} is out of bounds.")
                return None
        else:
            print("Column 'Valid_From_UTC' not found in the DataFrame.")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_data_stats(data):
    return data.describe()

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
    print(f"Saving results to:\n{file_path}")


def calculate_mse(actual_values, predicted_values):
  cum_sum_of_errors = 0
  n = len(predicted_values)
  for (actual,expected) in zip(actual_values,predicted_values):
    error = (expected - actual) ** 2
    cum_sum_of_errors += error
  
  mse = cum_sum_of_errors / n
  return mse

def calculate_log_likelihood(y_true, y_pred, sigma):
    """
    Calculate the log likelihood of the predicted values given the true values.
    
    The log likelihood is calculated using the formula:
    
    log_likelihood = -0.5 * (n * log(2 * pi) + n * log(variance) + sum((y_true - y_pred) ** 2) / variance)
    
    :param y_true: array-like, true values
    :param y_pred: array-like, predicted values
    :return: float, log likelihood
    """
    all_same_length = len(y_true) == len(y_pred) == len(sigma)
    if not all_same_length:
      raise ValueError("y_true, y_pred, and sigma must have the same length.")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if sigma is None:
      raise ValueError("Sigma must be provided for log likelihood calculation.")
      # raising an error instead of defaulting to make sure we don't assume uniform variance
      # variance = np.var(y_true)
    else:
      variances = np.asarray(sigma) ** 2
    
    n = len(y_true)
    
    log_likelihood = -0.5 * (np.sum(np.log(2 * np.pi * variances) + (y_true - y_pred) ** 2 / variances))
    
    return log_likelihood

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

def calculate_mae(y_true, y_pred):
  """
  Calculate the Mean Absolute Error (MAE).

  MAE = (1/n) * sum(|y_true - y_pred|)

  :param y_true: array-like, true values
  :param y_pred: array-like, predicted values
  :return: float, mean absolute error
  """
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  return np.mean(np.abs(y_true - y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
  """
  Calculate Mean Absolute Percentage Error (MAPE).

  MAPE = (100/n) * sum(|y_true - y_pred| / |y_true|)

  Note: Excludes samples where y_true is zero to avoid division by zero.

  :param y_true: array-like, true values
  :param y_pred: array-like, predicted values
  :return: float, mean absolute percentage error
  """
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  # Filter out zero values in y_true to avoid division by zero
  non_zero_mask = y_true != 0
  if not np.any(non_zero_mask):
    return np.nan
  y_true_filtered = y_true[non_zero_mask]
  y_pred_filtered = y_pred[non_zero_mask]
  return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100

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

  ledger_mse = np.array([])
  ledger_nmse = np.array([])
  ledger_mae = np.array([])
  ledger_mape = np.array([])
  ledger_logl = np.array([])
  ledger_var_actual_values = np.array([])
  ledger_var_predictions = np.array([])

  if algo == "chronos_2":
    # Initialize Chronos-2 pipeline
    print("= = = = >")
    print("Initializing Chronos-2 pipeline...\n")
    pipeline = Chronos2Pipeline.from_pretrained(
      "amazon/chronos-2",
      device_map=DEVICE_MAP,
      dtype=torch.bfloat16,
    )
  elif algo.startswith("chronos_") or algo.startswith("chronos-"):
    valid_chronos_sizes = {"tiny", "mini", "small", "base", "large"}

    if algo.startswith("chronos_"):
      # Extract the part after "chronos_"
      suffix = algo[len("chronos_"):]

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

    elif "-" in algo:
      print("= = = = >")
      print(f"Initializing CUSTOM Chronos {algo} pipeline...\n")
      pipeline = ChronosPipeline.from_pretrained(
      f"froyoresearcher/{algo}",
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
      algo_sigma = None

      # Summon designated algorithm
      if algo == "chronos_2":
        algo_predictions, algo_sigma = chronos_2_predict(df, column, context_start, context_finish, prediction_length=prediction_length, plot=plot, pipeline=pipeline)
      elif algo.startswith("chronos"):
        algo_predictions, algo_sigma = chronos_predict(df, column, context_start, context_finish, prediction_length, plot=plot, pipeline=pipeline)
      elif algo == "sarima":
        algo_predictions = sarima_predict(df,column,context_start, context_finish,prediction_length,plot=plot)
      elif algo == "arima":
        from wrappers.arima_wrapper import arima_predict
        algo_predictions, algo_sigma = arima_predict(df,column,context_start, context_finish,prediction_length,plot=plot)        
      elif algo == "gp":
          algo_predictions, algo_sigma = gp_predict(df,column, context_start, context_finish, prediction_length, plot=plot)
      else:
        raise ValueError(f"Invalid algorithm {algo}")
      
      # Obtain Actual Values
      forecast_start_index = context_finish
      forecast_end_index = forecast_start_index + prediction_length
      n_predictions = len(algo_predictions)

      actual_values = df[forecast_start_index:forecast_end_index][column]
      n_actual_values = len(actual_values)

      if not n_actual_values == n_predictions:
        raise ValueError(f"Unequal lengths of prediction and actual values ({n_actual_values} != {n_predictions})")
      
      # Calculate Mean Square Error
      mse = calculate_mse(actual_values, algo_predictions)
      ledger_mse = np.append(ledger_mse,mse)

      # Calculate Normalized Mean Square Error
      nmse = calculate_nmse(actual_values, algo_predictions)
      ledger_nmse = np.append(ledger_nmse,nmse)

      # Calculate Mean Absolute Error
      mae = calculate_mae(actual_values, algo_predictions)
      ledger_mae = np.append(ledger_mae, mae)

      # Calculate Mean Absolute Percentage Error
      mape = mean_absolute_percentage_error(actual_values, algo_predictions)
      ledger_mape = np.append(ledger_mape, mape)

      # Calculate Log Likelihood
      log_likelihood = calculate_log_likelihood(actual_values, algo_predictions, algo_sigma)
      ledger_logl = np.append(ledger_logl,log_likelihood)

      # Calculate Variance of Actual Values
      variance_actual_values = np.var(actual_values)
      ledger_var_actual_values = np.append(ledger_var_actual_values,variance_actual_values)

      # Calculate Variance of Predictions
      variance_predictions = np.var(algo_predictions)
      ledger_var_predictions = np.append(ledger_var_predictions,variance_predictions)

      if not np.isnan(mse) and not np.isnan(nmse):
        num_successful_runs += 1


  # Calculate elapsed time in seconds
  end_time = time.time()  
  elapsed_seconds = end_time - start_time

  # Convert elapsed time to hours
  elapsed_hours = elapsed_seconds / 3600


  # Calculate dataset statistics
  dataset_length = len(df)
  dataset_variance = np.var(df[column])
  data_set_mean = np.mean(df[column])


  # Mean MSE, NMSE, MAE, MAPE
  mean_mse = np.mean(ledger_mse[~np.isnan(ledger_mse)])
  mean_nmse = np.mean(ledger_nmse[~np.isnan(ledger_nmse)])
  mean_mae = np.mean(ledger_mae[~np.isnan(ledger_mae)])
  mean_mape = np.mean(ledger_mape[~np.isnan(ledger_mape)])
  median_mape = np.median(ledger_mape[~np.isnan(ledger_mape)])

  # Save results in a txt file
  algo_results = {
    "algorithm": algo,
    "data_title": data_title,
    "dataset_length": dataset_length,
    "dataset_mean": data_set_mean,
    "dataset_variance": dataset_variance,
    "context_length": context_length,
    "prediction_length": prediction_length,
    "successful_run_percentage": num_successful_runs / num_possible_iterations * 100,
    "mean_mse": mean_mse,
    "mean_nmse": mean_nmse,
    "mean_mae": mean_mae,
    "mean_mape": mean_mape,
    "median_mape":median_mape,
    "elapsed_hours": elapsed_hours,
    "num_possible_iterations": num_possible_iterations,
    "num_successful_runs": num_successful_runs,
    "ledger_mse": ledger_mse,
    "ledger_nmse": ledger_nmse,
    "ledger_mae": ledger_mae,
    "ledger_mape": ledger_mape,
    "ledger_logl": ledger_logl,      
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
  file_name = f"{RESULTS_FOLDER_NAME}/{algo}/data_title|{data_title}|cl|{context_length}|pl|{prediction_length}|algo|{algo}|ts|{timestamp}.txt"
  save_dict_to_json(algo_results, file_name)

  return algo_results

def compare_prediction_methods(df, data_column, date_column, context_start, context_finish, prediction_length, plot=True,methods=["chronos_mini","arima","sarima","gp","lstm"], run_name=None):
  # convert dates to an integer index
  if type(context_start) == str:
    context_start = find_first_occurrence_index(df, context_start, date_column)
    context_finish = find_first_occurrence_index(df, context_finish, date_column)

  # Determine Forecast Indices
  forecast_start_index = context_finish
  forecast_end_index = forecast_start_index + prediction_length
  
  joint_results = {}
  for method in methods:
    if method == "chronos_2":
      chronos_2_predictions, chronos_2_sigma = chronos_2_predict(df, data_column, context_start, context_finish, prediction_length=prediction_length, plot=plot, run_name=run_name)
      joint_results[method] = {"predictions": chronos_2_predictions, "sigma": chronos_2_sigma}
    elif method.startswith("chronos"):
      chronos_predictions, chronos_sigma = chronos_predict(df, data_column, context_start,context_finish, prediction_length, plot=plot, version=method, run_name=run_name)
      joint_results[method] = {"predictions":chronos_predictions, "sigma":chronos_sigma}
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
    elif method == "arima":
      arima_predictions, arima_sigma = arima_predict(df, data_column, context_start, context_finish, prediction_length, plot=plot, run_name=run_name)
      joint_results[method] = {"predictions":arima_predictions, "sigma":arima_sigma}
    elif method == "gp":
      gp_predictions, gp_sigma = gp_predict(df, data_column, context_start, context_finish, prediction_length, plot=plot, run_name=run_name)
      joint_results[method] = {"predictions":gp_predictions, "sigma":gp_sigma}
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