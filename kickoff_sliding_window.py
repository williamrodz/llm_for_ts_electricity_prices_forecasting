import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import json
from constants import *


# - - - - - - - - - - - - - - - - - - 
# System prices - START
# 
# 

date_column = "Date"

system_prices = pd.read_excel("data/electricitypricesdataset270624.xlsx", sheet_name="Data")
system_prices[date_column] = pd.to_datetime(system_prices[date_column])
system_prices.set_index(date_column)

start = '2022-01-01'
end = '2023-01-01'
column = 'Daily average'
prediction_length = 64

context_window_length = utils.find_first_occurrence_index(system_prices,end,date_column) - utils.find_first_occurrence_index(system_prices,start,date_column)
# print(f"context_window_length: {context_window_length}")
# results = utils.sliding_window_analysis_for_algorithm("chronos_small","Daily System Prices", system_prices,column,context_window_length,prediction_length)

# System prices - END

# - - - - - - - - - - - - - - - - - - 
#  Half hourly electricity prices
# 
# 

half_hourly_prices = pd.read_csv(f"{DATA_FOLDER}/Agile_Octopus_C_London-AGILE-22-07-22.csv")
data_column = "Price_Ex_VAT"
context_window_length = 7 * 48
prediction_length = 48
results = utils.sliding_window_analysis_for_algorithm("sarima","Half Hourly Prices", half_hourly_prices[:1000],data_column,context_window_length,prediction_length)


# def kickoff_sliding_window(algorithm_name, data_file_name, column_name, date_column, context_window_length, prediction_length, sheet_name=None):
#     if data_file_name.endswith(".csv"):
#         df = pd.read_csv(data_file_name)
#     elif data_file_name.endswith(".xlsx"):
#         df = pd.read_excel(data_file_name, sheet_name=sheet_name)
#     else:
#         raise ValueError("Data file must be a CSV or Excel file")


#     results = utils.sliding_window_analysis_for_algorithm(algorithm_name, data_file_name, df, column_name,context_window_length,prediction_length)
#     return results

# if __name__ == "__main__":
#     algorithm_name = "chronos_small"
#     data_file_name = "data/electricitypricesdataset270624.xlsx"
#     column_name = "Daily average"
#     date_column = "Date"
#     context_window_length = 20
#     prediction_length = 64
#     sheet_name = "Data"
#     results = kickoff_sliding_window(algorithm_name, data_file_name, column_name, date_column, context_window_length, prediction_length, sheet_name)
#     print(results)
