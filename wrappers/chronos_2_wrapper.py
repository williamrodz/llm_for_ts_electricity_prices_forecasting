import torch
import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc

from constants import *


def chronos_2_predict(
    input_data: pd.DataFrame,
    column: str,
    context_start_index: int,
    context_end_index: int,
    context_slice: pd.Series = None,
    prediction_length: int = 48,
    pipeline=None,
    plot=True,
    run_name=None
):
    """
    Generate predictions using the Chronos-2 foundational model.

    Args:
        input_data: DataFrame containing the time series data
        column: Column name in the DataFrame to predict
        context_start_index: Start index of the context window
        context_end_index: End index of the context window
        context_slice: Optional pre-sliced context data (unused, for API compatibility)
        prediction_length: Number of steps to forecast
        pipeline: Pre-initialized Chronos2Pipeline (optional, for efficiency)
        plot: Whether to generate visualization
        run_name: Optional name for saving plots

    Returns:
        median_predictions: Array of predicted median values
        std_devs: Array of standard deviations for uncertainty estimation
    """

    # Initialize pipeline if not provided
    if pipeline is None:
        print("= = = > Chronos-2 pipeline not initialized. Loading amazon/chronos-2...")
        pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map=DEVICE_MAP
        )

    # Determine size of input time series
    n = len(input_data)

    if isinstance(context_start_index, str) or isinstance(context_end_index, str):
        raise ValueError("Both context_start_index and context_end_index must be integers")

    # Build context dataframe with required columns for Chronos-2
    # Chronos-2 expects: id, timestamp, and target columns with REGULAR frequency
    # Use actual timestamps if available, otherwise create synthetic ones

    # Apply id column value to all rows
    context_dataframe = input_data.copy()
    context_dataframe.loc[:, 'id'] = 0
    # Rename column as 'target'
    context_dataframe = context_dataframe.rename(columns={column: 'target'})

    # Extract context slice
    context_slice_df = context_dataframe[context_start_index:context_end_index]
    length_of_context_slice = len(context_slice_df)

    # Create future dataframe with covariates and without the target column
    future_dataframe = context_dataframe.copy()
    future_dataframe = future_dataframe[context_end_index:context_end_index + prediction_length]
    future_dataframe.drop(columns=['target'], inplace=True)
    # Print the columns names of the future dataframe
    # print("Future DataFrame Columns")
    # print(future_dataframe.columns)
    # print("Context Slice DataFrame Tail")
    # print(context_slice_df.tail())
    # print("Future DataFrame Head")
    # print(future_dataframe.head())

    # Generate predictions using Chronos-2
    # Returns DataFrame with quantile predictions
    pred_df = pipeline.predict_df(
        context_slice_df,
        # future_df=future_dataframe,
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    # Extract quantiles from the prediction DataFrame
    # Chronos-2 returns columns: 'predictions' for median, '0.025', '0.975' for quantiles
    low = pred_df['0.1'].values
    median = pred_df['predictions'].values
    high = pred_df['0.9'].values

    median_predictions = median
    low_predictions = low
    high_predictions = high

    # Estimate standard deviation from the 95% prediction interval
    # For a normal distribution: 95% CI = mean +/- 1.96*std
    # So std ~= (high - low) / (2 * 1.96)
    std_devs = (high - low) / (2 * 1.96)

    num_median_predictions = len(median_predictions)

    # Plot a window of actual data two prediction windows to each side
    left_most_index = max(0, context_start_index - 2 * prediction_length)
    right_most_index = min(n, context_end_index + 2 * prediction_length)

    if plot:
        import utils  # Lazy import to avoid circular dependency

        plt.figure(figsize=(8, 4))
        rc('font', size=15)

        # Map time steps to corresponding dates
        reference_dates = [utils.map_timestep_to_date(idx) for idx in range(left_most_index, context_start_index)]
        context_dates = [utils.map_timestep_to_date(idx) for idx in range(context_start_index, context_end_index)]
        future_dates = [utils.map_timestep_to_date(idx) for idx in range(context_end_index, right_most_index)]
        prediction_dates = [utils.map_timestep_to_date(idx) for idx in range(context_end_index, context_end_index + num_median_predictions)]

        # Combine all dates together to calculate tick positions
        all_dates = reference_dates + context_dates + future_dates

        # Calculate positions for 0%, 25%, 50%, 75%, and 100% of the date range
        num_dates = len(all_dates)
        ticks_positions = [
            all_dates[0],
            all_dates[num_dates // 4],
            all_dates[num_dates // 2],
            all_dates[3 * num_dates // 4],
            all_dates[-1]
        ]

        # Plotting with dates on the x-axis
        plt.plot(reference_dates, input_data[column][left_most_index:context_start_index], color="royalblue", label="Historical Data")
        plt.plot(context_dates, input_data[column][context_start_index:context_end_index], color="green", label="Context Data")
        plt.plot(future_dates, input_data[column][context_end_index:right_most_index], color="royalblue")
        plt.plot(prediction_dates, median_predictions, color="tomato", label="Chronos-2 Median Forecast")
        plt.fill_between(prediction_dates, low_predictions, high_predictions, color="tomato", alpha=0.3, label="95% Prediction Interval")

        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Price per kWh (pence)")
        plt.grid()

        # Set custom x-ticks at the calculated positions
        plt.gca().set_xticks(ticks_positions)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=0)

        if run_name and isinstance(run_name, str) and run_name != "":
            plt.savefig(f"results/plots/chronos_2_{run_name}.png", dpi=300)

    return median_predictions, std_devs
