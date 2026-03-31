import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

load_dotenv()

# -----------------------------
# Config
# -----------------------------
API_KEY      = os.environ.get("ENTSOE_KEY", "")
DEFAULT_AREA = "NO_2"          # Norway zone 2 (major hub); see AREAS below
START_DATE   = "2015-01-05"    # Earliest data on ENTSO-E Transparency Platform
WINDOW_DAYS  = 365
OUTPUT_DIR   = Path("data")

# Nord Pool / ENTSO-E bidding zones available via the free API
AREAS = {
    # Norway (5 zones)
    "NO_1": "Norway NO1",
    "NO_2": "Norway NO2",
    "NO_3": "Norway NO3",
    "NO_4": "Norway NO4",
    "NO_5": "Norway NO5",
    # Sweden (4 zones)
    "SE_1": "Sweden SE1",
    "SE_2": "Sweden SE2",
    "SE_3": "Sweden SE3",
    "SE_4": "Sweden SE4",
    # Denmark
    "DK_1": "Denmark DK1",
    "DK_2": "Denmark DK2",
    # Finland
    "FI":   "Finland",
    # Baltic states
    "EE":   "Estonia",
    "LV":   "Latvia",
    "LT":   "Lithuania",
    # Other European markets (also on ENTSO-E)
    "DE_LU": "Germany/Luxembourg",
    "FR":    "France",
    "NL":    "Netherlands",
    "BE":    "Belgium",
    "AT":    "Austria",
    "GB":    "Great Britain",
}

OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# Helper functions
# -----------------------------
def date_windows(start: date, end: date, window: int = WINDOW_DAYS):
    """Split a date range into chunks of at most `window` days."""
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=window - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def fetch_window(client: EntsoePandasClient, area: str, w_from: date, w_to: date) -> pd.DataFrame:
    """Fetch day-ahead prices for one date window. Returns a two-column DataFrame."""
    start_ts = pd.Timestamp(datetime(w_from.year, w_from.month, w_from.day), tz="UTC")
    end_ts   = pd.Timestamp(datetime(w_to.year,   w_to.month,   w_to.day, 23, 59), tz="UTC")

    series = client.query_day_ahead_prices(area, start=start_ts, end=end_ts)

    df = series.rename("price_eur_mwh").reset_index()
    df.columns = ["timestamp_utc", "price_eur_mwh"]
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def download_area(area: str, start_date: str, end_date: str) -> None:
    """Download all day-ahead prices for a bidding zone, chunked into yearly windows."""
    if not API_KEY:
        raise ValueError("ENTSOE_KEY environment variable is not set.")

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = datetime.strptime(end_date,   "%Y-%m-%d").date()

    label = AREAS.get(area, area)
    windows = list(date_windows(start, end))

    print(f"Downloading ENTSO-E day-ahead prices — {label} ({area})")
    print(f"  {start_date} → {end_date}  |  {len(windows)} window(s)\n")

    client = EntsoePandasClient(api_key=API_KEY)
    all_frames = []

    for w_from, w_to in tqdm(windows, desc="Windows", unit=" yr"):
        try:
            df = fetch_window(client, area, w_from, w_to)
            if not df.empty:
                all_frames.append(df)
        except Exception as e:
            print(f"\n  Warning: window {w_from} → {w_to} failed: {e}")

    if not all_frames:
        print("No data returned.")
        return

    result = (pd.concat(all_frames, ignore_index=True)
                .drop_duplicates(subset=["timestamp_utc"])
                .sort_values("timestamp_utc")
                .reset_index(drop=True))

    first = result["timestamp_utc"].iloc[0].strftime("%m%y")
    last  = result["timestamp_utc"].iloc[-1].strftime("%m%y")
    output_file = OUTPUT_DIR / f"day_ahead_nordpool_{area.lower()}_{first}_{last}.csv"

    result.to_csv(output_file, index=False)

    print(f"\nSaved {len(result):,} rows to {output_file}")
    print(f"  First: {result['timestamp_utc'].iloc[0]}")
    print(f"  Last:  {result['timestamp_utc'].iloc[-1]}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download ENTSO-E day-ahead prices for Nord Pool bidding zones.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available areas:\n" + "\n".join(f"  {k:8s} {v}" for k, v in AREAS.items()),
    )
    parser.add_argument("--area",  type=str, default=DEFAULT_AREA,
                        help=f"Bidding zone code (default: {DEFAULT_AREA})")
    parser.add_argument("--start", type=str, default=START_DATE,
                        help=f"Start date YYYY-MM-DD (default: {START_DATE})")
    parser.add_argument("--end",   type=str, default=date.today().isoformat(),
                        help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    if args.area not in AREAS:
        print(f"Warning: '{args.area}' not in known AREAS list — attempting anyway.")

    download_area(args.area, args.start, args.end)


if __name__ == "__main__":
    main()
