import os

# Use only 1 GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline, Chronos2Pipeline

# Load the Chronos-2 pipeline
# GPU recommended for faster inference, but CPU is also supported using device_map="cpu"
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")


# Load data as a long-format pandas data frame
context_df = pd.read_csv("data/pr_grid_load_data.csv")
context_df = context_df.sort_values("timestamp")
print("Input dataframe shape:", context_df.shape)
print(context_df.head())
# Sort df by timestamp
