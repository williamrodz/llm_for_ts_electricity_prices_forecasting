"""
download_gb_prices.py
=====================
Download GB day-ahead electricity prices from the Elexon Insights API
(Market Index Data / MID dataset, provider APXMIDP).

API documentation : https://bmrs.elexon.co.uk/api-documentation/endpoint/datasets/MID
Base URL          : https://data.elexon.co.uk/bmrs/api/v1/datasets/MID
Authentication    : None required (public, no API key)
Settlement periods: 48 half-hourly periods per day (30-min resolution)
Units             : £/MWh
Data available    : ~2018-01-01 onwards

The script fetches data in configurable windows (default 7 days), saves each
window to data/raw/gb_APXMIDP/{from}_{to}.csv, then merges everything into
data/day_ahead_gb_APXMIDP.csv.  Already-downloaded windows are skipped so the
script can be re-run to top up to today without re-fetching old data.

Usage examples
--------------
    # Download everything from default start date to today
    python download_gb_prices.py

    # Custom date range
    python download_gb_prices.py --start 2022-01-01 --end 2024-12-31

    # Larger fetch window (still safe; API has no hard documented window limit)
    python download_gb_prices.py --window 30
"""

import argparse
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL        = "https://data.elexon.co.uk/bmrs/api/v1/datasets/MID"
DATA_PROVIDER   = "APXMIDP"   # APX/EPEX day-ahead market (the active provider)
#   N2EXMIDP is also in the feed but returns price=0 / volume=0 for all periods
#   in the data observed, so APXMIDP is the one to use for actual prices.

START_DATE      = "2018-01-01"   # earliest date with confirmed data
WINDOW_DAYS     = 7              # days per API request (safe; no documented limit)
                                 # A 14-day window returned HTTP 400 in testing;
                                 # 7 days is reliable.
RETRY_COUNT     = 3
RETRY_WAIT      = 5              # seconds between retries

OUTPUT_DIR      = Path("data")
RAW_DIR         = Path("data/raw")
OUTPUT_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def date_windows(start: date, end: date, window: int = WINDOW_DAYS):
    """Yield (from_date, to_date) tuples of at most `window` days each."""
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=window - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def fetch_window(from_date: date, to_date: date) -> pd.DataFrame:
    """
    Fetch all MID records for the given date window.

    The API filters on startTime (UTC) using the ``from`` and ``to`` query
    parameters.  startTime for settlement period 1 of a given settlement date
    is midnight UTC (00:00:00Z).  The ``to`` date is treated as an exclusive
    upper bound at midnight UTC on that date, meaning period 1 of ``to`` IS
    included but nothing beyond it.

    To guarantee all 48 settlement periods for every day in [from_date,
    to_date], we request ``to = to_date + 1 day`` and then filter the response
    client-side to keep only records with settlementDate <= to_date.

    The API returns all matching records in a single response (no pagination
    envelope).  A 7-day window yields ~336 records (48 x 7).

    Parameters
    ----------
    from_date : date
        First settlement date to include (inclusive).
    to_date : date
        Last settlement date to include (inclusive).

    Returns
    -------
    pd.DataFrame
        Columns: dataset, startTime, dataProvider, settlementDate,
                 settlementPeriod, price, volume
    """
    # Request one extra day so the API includes all 48 periods of to_date
    api_to = to_date + timedelta(days=1)

    params = {
        "from":           from_date.isoformat(),
        "to":             api_to.isoformat(),
        "dataProviders":  DATA_PROVIDER,
        "format":         "json",
    }

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            records = payload.get("data", [])
            if not records:
                return pd.DataFrame()
            # Filter client-side: keep only the requested settlement dates
            to_str = to_date.isoformat()
            from_str = from_date.isoformat()
            records = [r for r in records
                       if from_str <= r.get("settlementDate", "") <= to_str]
            return pd.DataFrame(records)

        except requests.HTTPError as e:
            if resp.status_code == 429:
                wait = 60 * attempt
                tqdm.write(f"  Rate-limited — waiting {wait}s (attempt {attempt}/{RETRY_COUNT})")
                time.sleep(wait)
            elif attempt < RETRY_COUNT:
                tqdm.write(f"  HTTP {resp.status_code} on attempt {attempt}, retrying in {RETRY_WAIT}s…")
                time.sleep(RETRY_WAIT)
            else:
                raise RuntimeError(f"HTTP error after {RETRY_COUNT} attempts: {e}") from e
        except Exception as e:
            if attempt < RETRY_COUNT:
                tqdm.write(f"  Error on attempt {attempt}: {e}. Retrying in {RETRY_WAIT}s…")
                time.sleep(RETRY_WAIT)
            else:
                raise RuntimeError(f"Failed after {RETRY_COUNT} attempts: {e}") from e

    raise RuntimeError("fetch_window: retry loop exhausted unexpectedly")


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names and parse timestamps.

    The raw API response has these fields:
        dataset, startTime, dataProvider, settlementDate,
        settlementPeriod, price, volume

    We rename startTime -> timestamp_utc and parse it as UTC-aware datetime.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "startTime" in df.columns:
        df = df.rename(columns={"startTime": "timestamp_utc"})

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Keep only the columns relevant for price forecasting work
    keep = ["timestamp_utc", "settlementDate", "settlementPeriod", "price", "volume",
            "dataProvider"]
    df = df[[c for c in keep if c in df.columns]]

    return df.sort_values("timestamp_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main download routine
# ---------------------------------------------------------------------------

def download(start_date: str, end_date: str, window: int = WINDOW_DAYS) -> None:
    """
    Download GB day-ahead prices for the given date range.

    Each window is saved to data/raw/gb_APXMIDP/{from}_{to}.csv immediately.
    All windows are merged and de-duplicated into
        data/day_ahead_gb_APXMIDP.csv
    on completion.
    """
    raw_dir = RAW_DIR / f"gb_{DATA_PROVIDER}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = datetime.strptime(end_date,   "%Y-%m-%d").date()

    windows = list(date_windows(start, end, window))
    print(f"Downloading GB day-ahead prices ({DATA_PROVIDER}) {start_date} → {end_date}")
    print(f"  {len(windows)} window(s) of up to {window} day(s) each\n")

    for w_from, w_to in tqdm(windows, desc="Windows", unit="win"):
        raw_file = raw_dir / f"{w_from}_{w_to}.csv"

        if raw_file.exists():
            tqdm.write(f"  Skipping {w_from} → {w_to} (already on disk)")
            continue

        df = fetch_window(w_from, w_to)

        if df.empty:
            tqdm.write(f"  Warning: no data for {w_from} → {w_to}")
            continue

        df = normalise(df)
        df.to_csv(raw_file, index=False)
        tqdm.write(f"  Saved {len(df):,} rows → {raw_file.name}")

    # --- Merge all raw windows ---------------------------------------------------
    raw_files = sorted(raw_dir.glob("*.csv"))
    if not raw_files:
        print("No data downloaded.")
        return

    print(f"\nMerging {len(raw_files)} file(s)…")
    merged = pd.concat(
        [pd.read_csv(f, parse_dates=["timestamp_utc"]) for f in raw_files],
        ignore_index=True,
    )
    merged["timestamp_utc"] = pd.to_datetime(merged["timestamp_utc"], utc=True)
    merged = (merged
              .drop_duplicates(subset=["timestamp_utc"])
              .sort_values("timestamp_utc")
              .reset_index(drop=True))

    first = merged["timestamp_utc"].iloc[0].strftime("%m%y")
    last  = merged["timestamp_utc"].iloc[-1].strftime("%m%y")
    output_file = OUTPUT_DIR / f"day_ahead_gb_{DATA_PROVIDER}_{first}_{last}.csv"

    merged.to_csv(output_file, index=False)
    print(f"Saved {len(merged):,} rows to {output_file}")
    print(f"  First: {merged['timestamp_utc'].iloc[0]}")
    print(f"  Last:  {merged['timestamp_utc'].iloc[-1]}")
    print(f"  Settlement periods: 48 per day (30-min, £/MWh)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download GB day-ahead electricity prices from Elexon Insights API."
    )
    parser.add_argument(
        "--start", type=str, default=START_DATE,
        help=f"Start date YYYY-MM-DD (default: {START_DATE})",
    )
    parser.add_argument(
        "--end", type=str, default=date.today().isoformat(),
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--window", type=int, default=WINDOW_DAYS,
        help=f"Days per API request window (default: {WINDOW_DAYS}; max safe: 7)",
    )
    args = parser.parse_args()

    download(args.start, args.end, args.window)


if __name__ == "__main__":
    main()
