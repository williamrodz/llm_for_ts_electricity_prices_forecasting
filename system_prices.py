import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import json
from constants import *

system_prices = pd.read_excel("data/electricitypricesdataset270624.xlsx", sheet_name="Data")
system_prices[DATE_COLUMN] = pd.to_datetime(system_prices[DATE_COLUMN])
system_prices.set_index(DATE_COLUMN)

start = '2022-01-01'
end = '2023-01-01'
column = 'Daily average'
prediction_length = 64

context_window_length = utils.find_first_occurrence_index(system_prices,end,DATE_COLUMN) - utils.find_first_occurrence_index(system_prices,start,"Date")
# print(f"context_window_length: {context_window_length}")
results = utils.sliding_window_analysis_for_algorithm("chronos_mini","Daily System Prices", system_prices,column,context_window_length,prediction_length)
