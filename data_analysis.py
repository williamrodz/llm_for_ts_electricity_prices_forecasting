# Analyze data from the results forlder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import json
from constants import *


def main():
    algorithms_to_analyze = ["chronos_mini"]
    data_to_analyze = ["agile_octopus_london_alpha_3_months", "agile_octopus_london_beta_3_months",
                       "agile_octopus_london_delta_3_months"]

    context_window_length = 7 * 48
    prediction_length = 48

    results = {}

    for algorithm in algorithms_to_analyze:
    


