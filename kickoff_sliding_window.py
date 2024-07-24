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
- Alpha : 0 -> 10,402 (217 days, 10,416 samples)
- Alpha : 0 -> 10,402 (217 days, 10,416 samples)
- Beta : 10,403 -> 20,805 (217 days, 10,416 samples)
- Gamma: 20,806 -> 31,208 (217 days, 10,416 samples)
'''
 
alpha = {
    "csv_title": "agile_octopus_london",
    "data_title": "agile_octopus_london_alpha_3_months",
    "subsection_start": 0,
    "subsection_end": 4465,
    "data_column": "Price_Ex_VAT",
    "context_window_length": 7 * 48,
    "prediction_length": 48
}

beta = {
    "csv_title": "agile_octopus_london",
    "data_title": "agile_octopus_london_beta_3_months",
    "subsection_start": 4465,
    "subsection_end": 8930,
    "data_column": "Price_Ex_VAT",
    "context_window_length": 7 * 48,
    "prediction_length": 48
    }

delta = {
    "csv_title": "agile_octopus_london",
    "data_title": "agile_octopus_london_delta_3_months",
    "subsection_start": 8930,
    "subsection_end": 13396,
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
    minimum_running_length = context_window_length + prediction_length
    #df_to_slide_on = df[:minimum_running_length]

    results = utils.sliding_window_analysis_for_algorithm(algorithm,data_title, df_to_slide_on,data_column,context_window_length,prediction_length)

if __name__ == "__main__":
    main()
# - - - - - - - - - - - - - - - - - - 
# System prices - START
# 
# 

# date_column = "Date"

# system_prices = pd.read_excel("data/electricitypricesdataset270624.xlsx", sheet_name="Data")
# system_prices[date_column] = pd.to_datetime(system_prices[date_column])
# system_prices.set_index(date_column)

# start = '2022-01-01'
# end = '2023-01-01'
# column = 'Daily average'
# prediction_length = 64

# context_window_length = utils.find_first_occurrence_index(system_prices,end,date_column) - utils.find_first_occurrence_index(system_prices,start,date_column)
# print(f"context_window_length: {context_window_length}")
# results = utils.sliding_window_analysis_for_algorithm("chronos_small","Daily System Prices", system_prices,column,context_window_length,prediction_length)

# System prices - END

# - - - - - - - - - - - - - - - - - - 
#  Half hourly electricity prices
# 
# 

# half_hourly_prices = pd.read_csv(f"{DATA_FOLDER}/Agile_Octopus_C_London-AGILE-22-07-22.csv")
# data_column = "Price_Ex_VAT"
# context_window_length = 7 * 48
# prediction_length = 48
#results = utils.sliding_window_analysis_for_algorithm("sarima","Half Hourly Prices", half_hourly_prices[:1000],data_column,context_window_length,prediction_length)

