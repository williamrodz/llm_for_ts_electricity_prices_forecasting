import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt

def investigate_results(algorithm_names, data_segment, ledger_key, run_name):
    alog_latex_label_map = {
      "chronos_tiny": "Chronos Tiny (8M params)",        
      "chronos_mini": "Chronos Mini (20M params)",
      "chronos_small": "Chronos Small (46M params)",
      "chronos_base": "Chronos Base (200M params)",
      "chronos_large": "Chronos Large (710M params)",
      "arima": "ARIMA",
      "gp": "Gaussian Process (Composite Kernel)",
      "chronos-tiny-336-48-8_000-alpha": "Chronos Tiny Fined Tuned on Alpha ",
      "chronos-tiny-336-48-8_000-beta": "Chronos Tiny Fine Tuned on Beta",
      "chronos-tiny-336-48-8_000-delta": "Chronos Tiny Fine Tuned on Delta",
      "chronos-tiny-336-48-8_000-abd": "Chronos Tiny Fine Tuned on ABD",
      # Add more algorithm mappings as needed
    }

    algorithm_color_map = {
        "chronos_tiny": "red",        
        "chronos_mini": "red",
        "chronos_small": "pink",
        "chronos_base": "pink",
        "chronos_large": "purple",
        "arima": "orange",
        "gp": "green",
        "chronos-tiny-336-48-8_000-alpha": "#2381c1",
        "chronos-tiny-336-48-8_000-beta": "#1c6eb4",
        "chronos-tiny-336-48-8_000-delta": "magenta",
        "chronos-tiny-336-48-8_000-abd": "black",
        # Add more algorithms and colors as needed
      }

    algorithm_linestyle_map = {
        "chronos_tiny": "-",        
        "chronos_mini": "--",
        "chronos_small": "-.",
        "chronos_base": "-",
        "chronos_large": "-",        
      "arima": '-',             # Dashed line
      "gp": '-',                # Dash-dot line
      "chronos-tiny-336-48-8_000-alpha": '-',      # Dotted line
      "chronos-tiny-336-48-8_000-beta": '-.',  # Dash-dot-dash pattern
      "chronos-tiny-336-48-8_000-delta": '-',       # Long dashes
      "chronos-tiny-336-48-8_000-abd": '-',   # Dash-dot-dash with longer gaps
      # Add more algorithms and linestyles as needed
    }

    ledger_key_map = {
      "ledger_mse": "Mean Squared Error",
      "ledger_nmse": "Normalized Mean Squared Error",
      "ledger_logl": "Log Likelihood",
      # Add more ledger keys as needed
    }

    plt.figure(figsize=(10, 6))
    plt.title(f'Comparison of {ledger_key_map[ledger_key]} Values for Different Algorithms on {data_segment.capitalize()} Segment')
    plt.xlabel('Sliding Window Iteration Number')
    plt.ylabel(ledger_key_map[ledger_key])

    if ledger_key == "ledger_nmse":
        plt.axhline(y=1, color='grey', linestyle='--', linewidth=1, label='Baseline (NMSE=1)')
    
    
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
        
        # Extract the ledger values
        if ledger_key not in data:
            print(f"Key {ledger_key} not found in {file_path}")
            continue
        
        ledger_values = np.array(data[ledger_key], dtype=float)
        
        # Plot the values
        alog_latex_label = alog_latex_label_map.get(algorithm_name, algorithm_name)
        color = algorithm_color_map.get(algorithm_name, "black")  # Default to black if not specified
        linestyle = algorithm_linestyle_map.get(algorithm_name, '-')  # Default to solid line if not specified
        plt.plot(ledger_values, label=alog_latex_label, color=color, linestyle=linestyle)
    
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/plots/{run_name}.png", dpi=300)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Investigate results in text files.')
    parser.add_argument('--algorithm_names', nargs='+', type=str, required=True, help='The names of the algorithms (space-separated list)')
    parser.add_argument('--data_segments', nargs='+', type=str, required=True, help='The data segments to look for (space-separated list)')
    parser.add_argument('--ledger_key', type=str, required=True, help='The ledger key to plot (e.g., ledger_mse, ledger_nmse, ledger_logl)')
    parser.add_argument('--run_name', type=str, required=True, help='The name of the run to save the plot')
    args = parser.parse_args()
    
    for data_segment in args.data_segments:
        investigate_results(args.algorithm_names, data_segment, args.ledger_key, args.run_name)