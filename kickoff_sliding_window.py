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

intended_data_dict = {
    "alpha": alpha,
    "beta": beta,
    "delta": delta
}

def main():

    parser = argparse.ArgumentParser(description="Process some data and run the algorithm.")

    # Add arguments for the algorithm and data
    parser.add_argument('-a', '--algorithm', type=str, required=True, help="Name of the algorithm to use")
    parser.add_argument('-d', '--data', type=str, required=True, help="Subsection of data to process")
    
    # Parse the arguments
    args = parser.parse_args()

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
    df_to_slide_on = df[subsection_start:subsection_end]

    # For debugging
    #minimum_running_length = context_window_length + prediction_length
    #extra_runs_for_debugging = 4
    #df_to_slide_on = df[:minimum_running_length + extra_runs_for_debugging]

    results = utils.sliding_window_analysis_for_algorithm(algorithm,data_title, df_to_slide_on,data_column,context_window_length,prediction_length)

if __name__ == "__main__":
    main()
