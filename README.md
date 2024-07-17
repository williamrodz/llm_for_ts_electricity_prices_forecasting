# Chronos Time Series Analysis

This repository contains scripts for analyzing time series energy data using a language model based method.

## Files Description

### requirements.txt 
This is an auto-generated file that lists all python package requirements for running the code in this project.

### chronos_wrapper.py
This file contains the code for using the Chronos language model for predicting TS data. 

### sarima_wrapper.py
Here you can find the code for generating forecast using the Seasonal - Auto Regressive Integrated Moving Average method. It is implemented using the `statsmodels.tsa.statespace.sarimax` package.

### lstm_wrapper.py
Code for predicting values using a LSTM based method. Data is normalized using a MinMaxScaler. A LSTM layer is followed by a linear layer. The LSTM layer processes the sequential input data (input_seq), maintaining an internal state (cell state and hidden state) that captures long-term dependencies in the data. During training, the model is trying to optimize MSE.

### gp_wrapper.py
Code for using a Gaussian Process based method for forecasting. Uses the `sklearn.gaussian_process` package.

### Mackey-Glass.py
This script generates a Mackey-Glass chaotic data series and saves it as a CSV file. The Mackey-Glass time series is a well-known chaotic dataset often used for testing time series forecasting models. You don't need to run it if  `mackey_glass_time_series.csv` already exists.

### mackey_glass_time_series.csv
This CSV file contains the generated Mackey-Glass chaotic time series data. It is used as input data for testing forecasting methods.

### system_price_overview.ipynb
See overview of system prices on different time horizons

### system_price_forecasting.ipynb
Compare performance of SARIMA, Chronos, LSTM, and Gaussian Processes forecasting on System Energy Price Data

### half_hourly_price_of_electricity
Compare performance of forecasting algorithms on half hourly energy prices of electricity. Provided by Octopus


### Data/

   Folder that contains all time series data for predicting and comparing performance of forecasting methods

## Getting Ready

To install the necessary dependencies for this project, follow these steps:

1. **Ensure you have Python and pip installed**: You can check if you have them installed by running the following commands in your terminal:

    ```sh
    python --version
    pip --version
    ```

2. **Navigate to the project directory**: Open your terminal and navigate to the directory where the `requirements.txt` file is located. For example:

    ```sh
    cd path/to/this/repo/oracle
    ```

3. **Install the dependencies**: Run the following command to install all the required packages listed in the `requirements.txt` file:

    ```sh
    pip install -r requirements.txt
    ```

After completing these steps, all the necessary packages specified in the `requirements.txt` file should be installed and ready to use.


## Usage

- Open and run the python notebooks listed above to see comparisons of forecasts of energy data 

