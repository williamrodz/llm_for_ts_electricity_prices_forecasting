import os
import json
import numpy as np
import argparse
from scipy.stats import ttest_rel, shapiro, f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def compute_statistics(mse_values):
    stats = {}
    for method, values in mse_values.items():
        mean_mse = np.mean(values)
        std_mse = np.std(values, ddof=1)
        median_mse = np.median(values)
        ci_lower = mean_mse - 1.96 * std_mse / np.sqrt(len(values))
        ci_upper = mean_mse + 1.96 * std_mse / np.sqrt(len(values))
        stats[method] = {
            "mean": mean_mse,
            "std": std_mse,
            "median": median_mse,
            "ci": (ci_lower, ci_upper),
        }
    return stats

def perform_tests(mse_values):
    values = list(mse_values.values())
    # Test for normality
    normality_results = {k: shapiro(v).pvalue for k, v in mse_values.items()}
    all_normal = all(p > 0.05 for p in normality_results.values())

    # Test for significant differences
    if all_normal:
        test_stat, p_value = f_oneway(*values)
        test_type = "ANOVA"
    else:
        test_stat, p_value = kruskal(*values)
        test_type = "Kruskal-Wallis"

    # Pairwise comparisons (Tukey's HSD if ANOVA, Dunn's test otherwise)
    combined_data = np.concatenate(values)
    groups = np.concatenate([[name] * len(vals) for name, vals in mse_values.items()])
    tukey_result = pairwise_tukeyhsd(combined_data, groups) if all_normal else None

    return {
        "normality": normality_results,
        "test_type": test_type,
        "p_value": p_value,
        "pairwise": tukey_result.summary() if tukey_result else None,
    }

# def generate_latex_table(statistics, test_results):
#     table = r"""\begin{table}[ht]
#             \centering
#             \caption{Statistical comparison of forecasting methods based on Mean Squared Error (MSE).}
#             \label{tab:mse_comparison}
#             \begin{tabular}{l S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] l}
#             \toprule
#             \textbf{Method} & \textbf{Mean MSE ± SD} & \textbf{Median MSE} & \textbf{95\% CI for Mean MSE} & \textbf{Significant Difference} \\
#             \midrule
#             """
#     # Add rows
#     for method, stats in statistics.items():
#         mean_sd = f"{stats['mean']:.3f} ± {stats['std']:.3f}"
#         median = f"{stats['median']:.3f}"
#         ci = f"[{stats['ci'][0]:.3f}, {stats['ci'][1]:.3f}]"
#         significance = "Yes" if test_results["p_value"] < 0.05 else "No"
#         table += f"{method} & {mean_sd} & {median} & {ci} & {significance} \\\\\n"
    
#     table += r"""\bottomrule
#             \end{tabular}
#             \end{table}"""
#     return table

def investigate_results(algorithm_names, data_segments):
    algo_latex_label_map = {
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
        mse_values = {}  # To collect all MSE values for significance testing
        mse_indeces_to_avoid = set([])  # To collect all clean MSE values for significance testing
        nmse_values = {}  # To collect all NMSE values for significance testing
        logl_values = {}  # To collect all Log-Likelihood values for significance testing

        

        for algorithm_name in algorithm_names:
            base_dir = "results"
            subfolder = os.path.join(base_dir, algorithm_name)
            file_pattern = f"agile_octopus_london_{data_segment}_2_weeks"
            
            files = [f for f in os.listdir(subfolder) if file_pattern in f]
            
            if not files:
                print(f"No file found for pattern {file_pattern} in algorithm {algorithm_name}")
                continue

            file_path = os.path.join(subfolder, files[0])
            
            with open(file_path, 'r') as file:
                data = json.load(file)

            for (i,mse_value) in enumerate(data['ledger_mse']):
                if np.isnan(mse_value):
                    mse_indeces_to_avoid.add(i)
            
            ledger_mse_mean = round(np.nanmean(data['ledger_mse']), 2)
            ledger_nmse_mean = round(np.nanmean(data['ledger_nmse']), 2)
            ledger_logl_median = round(np.nanmedian(data['ledger_logl']),2)

            algo_latex_label = algo_latex_label_map.get(algorithm_name, algorithm_name)
            
            results.append({
                'algorithm': algo_latex_label,
                'mean_mse': ledger_mse_mean,
                'mean_nmse': ledger_nmse_mean,
                'median_logl': ledger_logl_median
            })
                
            mse_values[algorithm_name] = np.array(data['ledger_mse'], dtype=float)
            nmse_values[algorithm_name] = np.array(data['ledger_nmse'], dtype=float)
            logl_values[algorithm_name] = np.array(data['ledger_logl'], dtype=float)

            # Check the normality of the data
            print(f"Algorithm: {algo_latex_label}")
            # print("Normality test for MSE If p < 0.05, normality is rejected.")
            _, p_value = shapiro(data['ledger_mse'])
            print(f"Normality is {'' if p_value < 0.05 else 'not'} rejected")
            print("Shapiro-Wilk p-value for Method 1:", p_value)

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
                algorithm = result['algorithm']
                mse = f"{result['mean_mse']:.2f}"
                nmse = f"{result['mean_nmse']:.2f}"
                logl = f"{result['median_logl']:.2f}"
                print(f"  {algorithm} & {mse} & {nmse} & {logl} \\\\")
                print("  \\hline")
            print("\\end{tabular}")
            print("\\caption{Algorithm Performance Across" + f" {data_set_label} " + "Dataset}")
            print("\\label{tab:methods_comparison_" + data_segment + "}")
            print("\\end{table}")
            print("\n")

            algorithm_names = list(mse_values.keys())
            #clean out the nan values
            for algorithm_name in algorithm_names:
                for i in (sorted(list((mse_indeces_to_avoid)), reverse=True)):
                    mse_values[algorithm_name] = np.delete(mse_values[algorithm_name], i)
                    # can do for nmse as well

            statistics = compute_statistics(mse_values)
            # print("MSE Statistics")
            for algo, stats in statistics.items():
                print(f"{algo_latex_label_map[algo]}: {stats}")

            test_results = perform_tests(mse_values)

            table = r"""\begin{table}[ht]
                    \centering
                    \caption{Statistical comparison of forecasting methods based on Mean Squared Error (MSE).}
                    \label{tab:mse_comparison}
                    \begin{tabular}{l S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] l}
                    \toprule
                    \textbf{Method} & \textbf{Mean MSE ± SD} & \textbf{Median MSE} & \textbf{95\% CI for Mean MSE} & \textbf{Significant Difference} \\
                    \midrule
                    """
            # Add rows
            for method, stats in statistics.items():
                mean_sd = f"{stats['mean']:.3f} ± {stats['std']:.3f}"
                median = f"{stats['median']:.3f}"
                ci = f"[{stats['ci'][0]:.3f}, {stats['ci'][1]:.3f}]"
                significance = "Yes" if test_results["p_value"] < 0.05 else "No"
                table += f"{algo_latex_label_map[method]} & {mean_sd} & {median} & {ci} & {significance} \\\\\n"
            
            table += r"""\bottomrule
                    \end{tabular}
                    \end{table}"""
            print()
            print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Investigate results in text files.')
    parser.add_argument('--algorithm_names', nargs='+', type=str, required=True, help='The names of the algorithms (space-separated list)')
    parser.add_argument('--data_segments', nargs='+', type=str, required=True, help='The data segments to look for (space-separated list)')

    args = parser.parse_args()
    
    investigate_results(args.algorithm_names, args.data_segments)