import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import json
from constants import *


def kickoff_sliding_window():
    csv_title = "agile_octopus_london"
    data_title = f"{csv_title}_alpha"
    subsection_start = 0
    subsection_end = 10416
    data_column = "Price_Ex_VAT"
    context_window_length = 7 * 48
    prediction_length = 48
    minimum_running_length = context_window_length + prediction_length # for debugging 

    df = pd.read_csv(f"{DATA_FOLDER}/{csv_title}.csv")
    df_to_slide_on = df[subsection_start:subsection_end]
    #df_to_slide_on = df[:minimum_running_length]


    #results = utils.sliding_window_analysis_for_algorithm("chronos_tiny",data_title, df_to_slide_on,data_column,context_window_length,prediction_length)
    results = utils.sliding_window_analysis_for_algorithm("gp",data_title, df_to_slide_on,data_column,context_window_length,prediction_length)
    # results = utils.sliding_window_analysis_for_algorithm("sarima",data_title, df_to_slide_on,data_column,context_window_length,prediction_length)

kickoff_sliding_window()
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

