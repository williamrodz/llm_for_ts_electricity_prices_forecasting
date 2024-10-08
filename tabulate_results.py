import os
import json
import numpy as np
import argparse

def investigate_results(algorithm_names, data_segments):
    alog_latex_label_map = {
      "chronos_tiny": "Chronos Tiny (8M params)",
      "chronos_mini": "Chronos Mini (20M params)",
      "chronos_small": "Chronos Small (46M params)",
      "chronos_base": "Chronos Base (200M params)",
      "chronos_large": "Chronos Large (710M params)",
      "arima": "ARIMA",
      "gp": "Gaussian Process (Composite Kernel)",
      "chronos-tiny-336-48-8_000-alpha": "Chronos Tiny (FT on Alpha)",
      "chronos-tiny-336-48-8_000-beta": "Chronos Tiny (FT on Beta)",
      "chronos-tiny-336-48-8_000-delta": "Chronos Tiny (FT on Delta)",
      "chronos-tiny-336-48-8_000-abd": "Chronos Tiny (FT on ABD Mix)",
    }
    
    for data_segment in data_segments:
        results = []

        for algorithm_name in algorithm_names:
            # Define the directory and filename pattern
            base_dir = "results"
            subfolder = os.path.join(base_dir, algorithm_name)
            file_pattern = f"agile_octopus_london_{data_segment}_2_weeks"
            
            # Find the file that matches the pattern
            files = [f for f in os.listdir(subfolder) if file_pattern in f]
            
            if not files:
                print(f"No file found for pattern {file_pattern} in algorithm {algorithm_name}")
                continue

            file_path = os.path.join(subfolder, files[0])
            
            # Read and parse the file as JSON
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            # Extract lists and calculate mean values, ignoring NaNs
            def calculate_mean(key):
                values = np.array(data[key], dtype=float)
                # print(values)
                return np.nanmean(values)
            
            def calculate_median(key):
                values = np.array(data[key], dtype=float)
                return np.nanmedian(values)

            ledger_mse_mean = round(calculate_mean('ledger_mse'), 2)
            ledger_nmse_mean = round(calculate_mean('ledger_nmse'), 2)
            ledger_logl_mean = round(calculate_median('ledger_logl'), 2)
            # print("ledger_logl_mean", ledger_logl_mean)

            alog_latex_label = alog_latex_label_map.get(algorithm_name, algorithm_name)
            
            results.append({
                'algorithm': alog_latex_label,
                'mean_mse': ledger_mse_mean,
                'mean_nmse': ledger_nmse_mean,
                'mean_logl': ledger_logl_mean
            })

        # Capitalize the data segment
        data_set_label = data_segment.capitalize()
        
        # Print the LaTeX code for the table
        if results:
            print("\n")
            print("\\begin{table}[h]")
            print("\\centering")
            print("\\begin{tabular}{|l|c|c|c|}")
            print("  \\hline")
            print("  \\textbf{Method} & \\textbf{Mean MSE} & \\textbf{Mean NMSE} & \\textbf{Median Log Likelihood} \\\\")
            print("  \\hline")
            for result in results:
                print(f"  {result['algorithm']} & {result['mean_mse']:.2f} & {result['mean_nmse']:.2f} & {result['mean_logl']:.2f} \\\\")
                print("  \\hline")
            print("\\end{tabular}")
            print("\\caption{Algorithm Performance Across" + f" {data_set_label} " + "Dataset}")
            # Add the label for the table, putting in the dataset name
            print("\\label{tab:methods_comparison_" + data_segment + "}")
            print("\\end{table}")
            print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Investigate results in text files.')
    parser.add_argument('--algorithm_names', nargs='+', type=str, required=True, help='The names of the algorithms (space-separated list)')
    parser.add_argument('--data_segments', nargs='+', type=str, required=True, help='The data segments to look for (space-separated list)')

    args = parser.parse_args()
    
    investigate_results(args.algorithm_names, args.data_segments)