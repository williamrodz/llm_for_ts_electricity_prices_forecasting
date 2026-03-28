import argparse
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# Config
# -----------------------------
PJM_API_URL  = "https://api.pjm.com/api/v1/da_hrl_lmps"
API_KEY      = os.environ.get("PJM_API_KEY", "")  # set via: export PJM_API_KEY=your_key
DEFAULT_NODE = 51291       # Western Hub — change or pass via --node
START_DATE   = "2018-01-01"
PAGE_SIZE    = 50_000
WINDOW_DAYS  = 365         # API max date range is 366 days
RETRY_COUNT  = 3
OUTPUT_DIR   = Path("data")

FIELDS = ",".join([
    "datetime_beginning_utc",
    "pnode_id",
    "pnode_name",
    "total_lmp_da",
    "system_energy_price_da",
    "congestion_price_da",
    "marginal_loss_price_da",
])

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


def _is_archive_error(resp: requests.Response) -> bool:
    """Return True if the 400 response is the PJM 'archived data' restriction."""
    try:
        errors = resp.json().get("errors", [])
        return any("archived data" in e.get("message", "").lower() for e in errors)
    except Exception:
        return False


def fetch_page(pnode_id: int, window_from: date, window_to: date, start_row: int) -> tuple[pd.DataFrame, int]:
    """
    Fetch one page of day-ahead LMP data via GET.
    Date range must be within 366 days.

    For historical (archived) windows the API rejects pnode_id, fields, sort, and
    order filters. In that case we fall back to a type=ZONE request (22 zone nodes,
    ~192K rows/year) and trim the response to the requested node client-side.
    NOTE: this fallback only works for zone-level pnode IDs.

    Returns (DataFrame, total_rows).
    """
    if not API_KEY:
        raise ValueError("PJM_API_KEY environment variable is not set.")

    # API expects: "yyyy-MM-dd HH:mm to yyyy-MM-dd HH:mm"
    date_range = f"{window_from:%Y-%m-%d} 00:00 to {window_to:%Y-%m-%d} 23:00"

    params = {
        "datetime_beginning_utc": date_range,
        "pnode_id":               pnode_id,
        "row_is_current":         "TRUE",
        "fields":                 FIELDS,
        "rowCount":               PAGE_SIZE,
        "startRow":               start_row,
        "order":                  "Asc",
        "sort":                   "datetime_beginning_ept",
    }
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = requests.get(PJM_API_URL, params=params, headers=headers, timeout=60)

            if resp.status_code == 429:
                wait = 30 * attempt
                print(f"\n  Rate limited — waiting {wait}s (attempt {attempt}/{RETRY_COUNT})...")
                time.sleep(wait)
                continue

            if resp.status_code == 400 and _is_archive_error(resp):
                # Archived windows reject pnode_id/fields/sort/order.
                # type=ZONE is still accepted and returns only the 22 zone nodes
                # (~192K rows/year vs ~104M unfiltered), then we trim client-side.
                archive_params = {
                    "datetime_beginning_utc": date_range,
                    "type":                   "ZONE",
                    "row_is_current":         "TRUE",
                    "rowCount":               PAGE_SIZE,
                    "startRow":               start_row,
                }
                resp = requests.get(PJM_API_URL, params=archive_params, headers=headers, timeout=60)
                resp.raise_for_status()
                j = resp.json()
                items = j.get("items", [])
                total = j.get("totalRows", len(items))
                df = pd.DataFrame(items)
                if not df.empty and "pnode_id" in df.columns:
                    df = df[df["pnode_id"] == pnode_id].reset_index(drop=True)
                return df, total

            if resp.status_code == 400:
                try:
                    err = resp.json().get("errors", resp.text)
                except Exception:
                    err = resp.text
                raise RuntimeError(f"400 Bad Request: {err}")

            resp.raise_for_status()
            j = resp.json()
            if "errors" in j:
                raise RuntimeError(f"API error: {j['errors']}")
            items = j.get("items", [])
            total = j.get("totalRows", len(items))
            return pd.DataFrame(items), total

        except RuntimeError:
            raise
        except Exception as e:
            if attempt < RETRY_COUNT:
                time.sleep(5)
            else:
                raise RuntimeError(f"Failed after {RETRY_COUNT} attempts: {e}")


def fetch_window(pnode_id: int, window_from: date, window_to: date) -> pd.DataFrame:
    """Fetch all pages for a single date window, handling pagination."""
    all_frames = []
    start_row  = 1

    while True:
        df, total = fetch_page(pnode_id, window_from, window_to, start_row)
        if df.empty:
            break
        all_frames.append(df)
        if start_row + len(df) - 1 >= total:
            break
        start_row += PAGE_SIZE

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


def download_node(pnode_id: int, start_date: str, end_date: str) -> None:
    """Download all day-ahead LMP data for a node, chunked into yearly windows."""
    output_file = OUTPUT_DIR / f"day_ahead_pjm_{pnode_id}.csv"

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = datetime.strptime(end_date,   "%Y-%m-%d").date()

    windows   = list(date_windows(start, end))
    all_frames = []

    print(f"Downloading PJM day-ahead LMPs — node {pnode_id} ({start_date} → {end_date})")
    print(f"  {len(windows)} yearly window(s) to fetch\n")

    for w_from, w_to in tqdm(windows, desc="Windows", unit=" yr"):
        df = fetch_window(pnode_id, w_from, w_to)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("No data returned.")
        return

    result = pd.concat(all_frames, ignore_index=True)

    # Normalise column names
    result.columns = [c.strip().lower() for c in result.columns]

    # Rename timestamp for clarity
    if "datetime_beginning_utc" in result.columns:
        result = result.rename(columns={"datetime_beginning_utc": "timestamp_utc"})

    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)

    # Deduplicate and sort
    result = (result
              .drop_duplicates(subset=["timestamp_utc"])
              .sort_values("timestamp_utc")
              .reset_index(drop=True))

    result.to_csv(output_file, index=False)

    print(f"\nSaved {len(result):,} rows to {output_file}")
    print(f"  First: {result['timestamp_utc'].iloc[0]}")
    print(f"  Last:  {result['timestamp_utc'].iloc[-1]}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Download PJM day-ahead LMP data by pricing node.")
    parser.add_argument("--node",  type=int, default=DEFAULT_NODE,
                        help=f"PJM pricing node ID (default: {DEFAULT_NODE})")
    parser.add_argument("--start", type=str, default=START_DATE,
                        help=f"Start date YYYY-MM-DD (default: {START_DATE})")
    parser.add_argument("--end",   type=str, default=date.today().isoformat(),
                        help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    download_node(args.node, args.start, args.end)


if __name__ == "__main__":
    main()
