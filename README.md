# Chronos Time Series Analysis

This repository contains scripts for generating and analyzing time series data using a large language model for time series (LLMTS).

## Files Description

### LLMTS.py
This script contains the main Chronos code. The LLMTS (Large Language Model for Time Series) is implemented here, which is used for analyzing and forecasting time series data.

### Mackey-Glass.py
This script generates a Mackey-Glass chaotic data series and saves it as a CSV file. The Mackey-Glass time series is a well-known chaotic dataset often used for testing time series forecasting models.

### mackey_glass_time_series.csv
This CSV file contains the generated Mackey-Glass chaotic time series data. It is used as input data for testing the LLMTS model.

### playground.py
This script is used for running the LLMTS code against specific time series datasets. Currently, it is configured to work with the Mackey-Glass time series data.

## Usage

1. **Generate Mackey-Glass Time Series:** (You can skip if `mackey_glass_time_series.csv` already exists)
   
   Run the `Mackey-Glass.py` script to generate the chaotic time series data.
   ```bash
   python Mackey-Glass.py
   ```


3. **Run LLMTS on Time Series Data:**
   Use the playground.py script to apply the LLMTS model to the Mackey-Glass time series data.
   ```bash
    python playground.py
   ```



