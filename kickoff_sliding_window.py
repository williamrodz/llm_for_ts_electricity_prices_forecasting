import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import json
from constants import *
import argparse

'''
Agile_Octopus_London.csv
- 31,208 time steps/ 1.78 years of per kwh, each representing a half hour price
- alpha: 0 -> 672 (2 weeks, 672 samples)
- beta: 672 -> 1344 (2 weeks, 672 samples)
- delta: 1344 -> 2016 (2 weeks, 672 samples)


'''
 
alpha = {
    "csv_title": "agile_octopus_london",
    "data_title": "agile_octopus_london_alpha_2_weeks",
    "subsection_start": 0,
    "subsection_end": 672,
    "data_column": "Price_Ex_VAT",
    "context_window_length": 7 * 48,
    "prediction_length": 48
}

beta = {
    "csv_title": "agile_octopus_london",
    "data_title": "agile_octopus_london_beta_2_weeks",
    "subsection_start": 672,
    "subsection_end": 1344,
    "data_column": "Price_Ex_VAT",
    "context_window_length": 7 * 48,
    "prediction_length": 48
    }

delta = {
    "csv_title": "agile_octopus_london",
    "data_title": "agile_octopus_london_delta_2_weeks",
    "subsection_start": 1344,
    "subsection_end": 2016,
    "data_column": "Price_Ex_VAT",
    "context_window_length": 7 * 48,
    "prediction_length": 48
    }

pr_grid_load_data = {
    "csv_title": "pr_grid_load_data",
    "data_title": "pr_grid_load_data",
    "subsection_start": "2026-01-01",
    "subsection_end": "2026-01-31",
    "data_column": "current_demand",
    "context_window_length": 7 * 48,
    "prediction_length": 48
}

intended_data_dict = {
    "alpha": alpha,
    "beta": beta,
    "delta": delta,
    "pr_grid": pr_grid_load_data
}

def generate_data_subsection_csvs():
    df = pd.read_csv(f"{DATA_FOLDER}/agile_octopus_london.csv")
    for key in intended_data_dict:
        config = intended_data_dict[key]
        subsection_start = config["subsection_start"]
        subsection_end = config["subsection_end"]
        csv_title = config["csv_title"]
        data_title = config["data_title"]
        data_column = config["data_column"]
        df_to_slide_on = df[subsection_start:subsection_end]
        df_to_slide_on.to_csv(f"{DATA_FOLDER}/{data_title}.csv", index=False)

def main():

    parser = argparse.ArgumentParser(description="Process some data and run the algorithm.")

    # Add arguments for the algorithm and data
    parser.add_argument('-a', '--algorithm', type=str, required=True, help="Name of the algorithm to use")
    parser.add_argument('-d', '--data', type=str, required=True, help="Subsection of data to process")
    parser.add_argument('-g', '--generate', action='store_true', help="Generate subsection csvs")

    # Parse the arguments
    args = parser.parse_args()

    if args.generate:
        generate_data_subsection_csvs()
        return
    

    # Extract the algorithm from arguments
    algorithm = args.algorithm
    intended_data = args.data

    config = intended_data_dict[intended_data]

    csv_title = config["csv_title"]
    data_title = config["data_title"]
    subsection_start = config["subsection_start"]
    subsection_end = config["subsection_end"]
    data_column = config["data_column"]
    context_window_length = config["context_window_length"]
    prediction_length = config["prediction_length"]

    df = pd.read_csv(f"{DATA_FOLDER}/{csv_title}.csv")

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    elif "Valid_From_UTC" in df.columns:
        # rename the column to timestamp
        df = df.rename(columns={"Valid_From_UTC": "timestamp"})
    else:
        raise ValueError("No timestamp column found in dataframe")
    
    # Add weekday column
    df = utils.add_weekday_column(df, "timestamp")

    # Filter by subsection - supports both int indices and date strings
    if isinstance(subsection_start, int) and isinstance(subsection_end, int):
        # Index-based filtering
        df_to_slide_on = df[subsection_start:subsection_end]
    elif isinstance(subsection_start, str) and isinstance(subsection_end, str):
        # Date-based filtering (inclusive start, exclusive end)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        start_dt = pd.to_datetime(subsection_start)
        end_dt = pd.to_datetime(subsection_end)
        df_to_slide_on = df[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)]
        df_to_slide_on = df_to_slide_on.reset_index(drop=True)
    else:
        raise ValueError("subsection_start and subsection_end must both be int or both be date strings")

    # Resample pr_grid data to regular intervals with gap filling
    if intended_data == "pr_grid":
        frequency_to_resample_to = '15min'
        df_to_slide_on = utils.resample_to_regular_intervals(
            df_to_slide_on,
            timestamp_column='timestamp',
            freq=frequency_to_resample_to,
            fill_gaps='interpolate'  # Required for Chronos-2 (needs regular timestamps)
        )
        print(f"Resampled to {frequency_to_resample_to} intervals with interpolation: {len(df_to_slide_on)} samples")
        #print(df_to_slide_on.head())
        # Check for gaps after resampling
        utils.check_timestamp_gaps(df_to_slide_on, timestamp_column='timestamp', expected_freq=frequency_to_resample_to)

    # For debugging
    #minimum_running_length = context_window_length + prediction_length
    #extra_runs_for_debugging = 4
    #df_to_slide_on = df[:minimum_running_length + extra_runs_for_debugging]

    results = utils.sliding_window_analysis_for_algorithm(algorithm,data_title, df_to_slide_on,data_column,context_window_length,prediction_length,subsection_start,subsection_end)

if __name__ == "__main__":
    main()
