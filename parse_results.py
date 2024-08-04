import os
import json
import numpy as np
import argparse

def investigate_results(algorithm_name, data_segment):
    # Define the directory and filename pattern
    base_dir = "results"
    subfolder = os.path.join(base_dir, algorithm_name)
    file_pattern = f"agile_octopus_london_{data_segment}_2_weeks"
    
    # Find the file that matches the pattern
    files = [f for f in os.listdir(subfolder) if file_pattern in f]
    print(f" Files found for pattern {file_pattern}: {files}")
    
    if not files:
        print(f"No file found for pattern {file_pattern}")
        return

    file_path = os.path.join(subfolder, files[0])
    
    # Read and parse the file as JSON
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Print all keys in the JSON
    print("Keys in JSON file:")
    for key in data.keys():
        print(key)
    
    # Extract lists and calculate mean values, ignoring NaNs
    def calculate_mean(key):
        values = np.array(data[key], dtype=float)
        return np.nanmean(values)

    ledger_mse_mean = calculate_mean('ledger_mse')
    ledger_nmse_mean = calculate_mean('ledger_nmse')
    ledger_logl_mean = calculate_mean('ledger_logl')

    alog_latex_label = None
    if algorithm_name == "chronos_mini":
      alog_latex_label = "Chronos Mini (20B params)"
    elif algorithm_name == "arima":
      alog_latex_label = "ARIMA"
    elif algorithm_name == "gp":
      alog_latex_label = "Gaussian Process"
    else:
       alog_latex_label = algorithm_name
    # Print the LaTeX code for the row
    latex_row = f"{alog_latex_label} & {ledger_mse_mean:.4f} & {ledger_nmse_mean:.4f} & {ledger_logl_mean:.4f} \\\\"
    print("")
    print("LaTeX row for table:")
    print("Method |           MSE | NMSE | Mean Log Likelihood ")
    print(latex_row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Investigate results in text files.')
    parser.add_argument('--algorithm_name', type=str, required=True, help='The name of the algorithm')
    parser.add_argument('--data_segment', type=str, required=True, help='The data segment to look for')

    args = parser.parse_args()
    
    investigate_results(args.algorithm_name, args.data_segment)