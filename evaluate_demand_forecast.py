import pandas as pd
import numpy as np
import argparse
from constants import DATA_FOLDER
from utils import calculate_mae, mean_absolute_percentage_error


def load_and_prepare_data(csv_path):
    """
    Load the grid load data CSV and prepare it for analysis.

    :param csv_path: Path to the CSV file
    :return: DataFrame sorted by timestamp with valid rows only
    """
    df = pd.read_csv(csv_path)

    # Convert timestamp to datetime (strip timezone, data is stored in UTC)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Drop rows with missing values in the columns we need
    df = df.dropna(subset=['current_demand', 'next_hour_demand_forecast'])

    return df


def evaluate_forecast_accuracy(df):
    """
    Evaluate the accuracy of next_hour_demand_forecast against current_demand.

    :param df: DataFrame with current_demand and next_hour_demand_forecast columns
    :return: Dictionary with error metrics
    """
    actual = df['current_demand'].values
    forecast = df['next_hour_demand_forecast'].values

    # Calculate error at each timestamp
    errors = actual - forecast
    abs_errors = np.abs(errors)

    # Calculate metrics
    mae = calculate_mae(actual, forecast)
    mape = mean_absolute_percentage_error(actual, forecast)

    # Additional statistics
    mean_error = np.mean(errors)  # Bias
    std_error = np.std(errors)
    max_abs_error = np.max(abs_errors)
    min_abs_error = np.min(abs_errors)

    results = {
        'num_samples': len(df),
        'mae': mae,
        'mape': mape,
        'mean_error_bias': mean_error,
        'std_error': std_error,
        'max_abs_error': max_abs_error,
        'min_abs_error': min_abs_error,
        'mean_actual_demand': np.mean(actual),
        'mean_forecast_demand': np.mean(forecast),
    }

    return results, errors


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate demand forecast accuracy from PR grid load data'
    )
    parser.add_argument(
        '-f', '--file',
        type=str,
        default=f'{DATA_FOLDER}/pr_grid_load_data.csv',
        help='Path to the CSV file (default: data/pr_grid_load_data.csv)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed per-timestamp errors'
    )
    parser.add_argument(
        '-s', '--start-date',
        type=str,
        default=None,
        help='Start date filter (inclusive), format: YYYY-MM-DD'
    )
    parser.add_argument(
        '-e', '--end-date',
        type=str,
        default=None,
        help='End date filter (exclusive), format: YYYY-MM-DD'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Demand Forecast Accuracy Evaluation")
    print("=" * 60)
    print(f"\nLoading data from: {args.file}")

    # Load and prepare data
    df = load_and_prepare_data(args.file)

    print(f"Loaded {len(df)} valid samples")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Apply date filters
    if args.start_date:
        start_dt = pd.to_datetime(args.start_date)
        df = df[df['timestamp'] >= start_dt]
        print(f"Filtered to start date >= {args.start_date}")

    if args.end_date:
        end_dt = pd.to_datetime(args.end_date)
        df = df[df['timestamp'] < end_dt]
        print(f"Filtered to end date < {args.end_date}")

    if args.start_date or args.end_date:
        print(f"Filtered samples: {len(df)}")
        if len(df) > 0:
            print(f"Filtered range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            print("No samples in the specified date range.")
            return None

    # Evaluate forecast accuracy
    results, errors = evaluate_forecast_accuracy(df)

    # Print results
    print("\n" + "-" * 60)
    print("Forecast Accuracy Metrics")
    print("-" * 60)
    print(f"Number of samples:         {results['num_samples']:,}")
    print(f"Mean Absolute Error (MAE): {results['mae']:.2f} MW")
    print(f"Mean Percentage Error:     {results['mape']:.2f}%")
    print(f"Mean Error (Bias):         {results['mean_error_bias']:.2f} MW")
    print(f"Std of Error:              {results['std_error']:.2f} MW")
    print(f"Max Absolute Error:        {results['max_abs_error']:.2f} MW")
    print(f"Min Absolute Error:        {results['min_abs_error']:.2f} MW")
    print("-" * 60)
    print(f"Mean Actual Demand:        {results['mean_actual_demand']:.2f} MW")
    print(f"Mean Forecast Demand:      {results['mean_forecast_demand']:.2f} MW")
    print("=" * 60)

    if args.verbose:
        print("\nPer-timestamp errors (first 20):")
        print("-" * 40)
        for i, (idx, row) in enumerate(df.head(20).iterrows()):
            error = row['current_demand'] - row['next_hour_demand_forecast']
            print(f"{row['timestamp']}: Actual={row['current_demand']:.0f}, "
                  f"Forecast={row['next_hour_demand_forecast']:.0f}, "
                  f"Error={error:.0f}")

    return results


if __name__ == "__main__":
    main()
