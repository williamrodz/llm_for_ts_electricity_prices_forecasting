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
API_KEY      = os.environ.get("PJM_API_KEY", "")
DEFAULT_NODE = 51291
START_DATE   = "2018-01-01"
PAGE_SIZE    = 50_000
WINDOW_DAYS  = 365         # API max date range is 366 days
RETRY_COUNT  = 3           # retries for non-rate-limit errors
MAX_429      = 5           # retries for rate limit (429)
OUTPUT_DIR   = Path("data")
RAW_DIR      = Path("data/raw")

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
RAW_DIR.mkdir(exist_ok=True)

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


def _get_with_retry(url: str, params: dict, headers: dict) -> requests.Response:
    """GET with exponential backoff on 429. Raises RuntimeError after MAX_429 attempts."""
    for attempt in range(1, MAX_429 + 1):
        resp = requests.get(url, params=params, headers=headers, timeout=60)
        if resp.status_code != 429:
            return resp
        wait = 60 * attempt
        print(f"\n  Rate limited — waiting {wait}s (attempt {attempt}/{MAX_429})...")
        time.sleep(wait)
    raise RuntimeError("Rate limit: all retries exhausted. Try again later.")


def _is_archive_error(resp: requests.Response) -> bool:
    """Return True if the 400 response is the PJM 'archived data' restriction."""
    try:
        errors = resp.json().get("errors", [])
        return any("archived data" in e.get("message", "").lower() for e in errors)
    except Exception:
        return False


def fetch_page(pnode_id: int, window_from: date, window_to: date, start_row: int) -> tuple[pd.DataFrame, int]:
    """
    Fetch one page of day-ahead LMP data.

    For historical (archived) windows the API rejects pnode_id, fields, sort, and
    order. Falls back to type=ZONE (22 zone nodes, ~192K rows/year) and filters
    client-side. Only works for zone-level pnode IDs.

    Returns (DataFrame, total_rows).
    """
    if not API_KEY:
        raise ValueError("PJM_API_KEY environment variable is not set.")

    date_range = f"{window_from:%Y-%m-%d} 00:00 to {window_to:%Y-%m-%d} 23:00"
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}

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

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = _get_with_retry(PJM_API_URL, params, headers)

            if resp.status_code == 400 and _is_archive_error(resp):
                # Archived windows reject pnode_id/fields/sort/order.
                # type=ZONE returns only the 22 zone nodes (~192K rows/year),
                # then we trim to the requested node client-side.
                archive_params = {
                    "datetime_beginning_utc": date_range,
                    "type":                   "ZONE",
                    "row_is_current":         "TRUE",
                    "rowCount":               PAGE_SIZE,
                    "startRow":               start_row,
                }
                resp = _get_with_retry(PJM_API_URL, archive_params, headers)
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

    raise RuntimeError("fetch_page: retry loop exhausted unexpectedly")


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


def _mmyy(dt: pd.Timestamp) -> str:
    return dt.strftime("%m%y")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and timestamp format."""
    df.columns = [c.strip().lower() for c in df.columns]
    if "datetime_beginning_utc" in df.columns:
        df = df.rename(columns={"datetime_beginning_utc": "timestamp_utc"})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def download_node(pnode_id: int, start_date: str, end_date: str) -> None:
    """
    Download all day-ahead LMP data for a node, chunked into yearly windows.

    Each window is saved to data/raw/pjm_{pnode_id}/{from}_{to}.csv immediately
    on success. Already-downloaded windows are skipped, enabling resumption
    after interruptions. All windows are merged into
    data/day_ahead_pjm_{pnode_id}_{MMYY}_{MMYY}.csv.
    """
    raw_node_dir = RAW_DIR / f"pjm_{pnode_id}"
    raw_node_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = datetime.strptime(end_date,   "%Y-%m-%d").date()

    windows = list(date_windows(start, end))
    print(f"Downloading PJM day-ahead LMPs — node {pnode_id} ({start_date} → {end_date})")
    print(f"  {len(windows)} yearly window(s) to fetch\n")

    for w_from, w_to in tqdm(windows, desc="Windows", unit=" yr"):
        raw_file = raw_node_dir / f"{w_from}_{w_to}.csv"

        if raw_file.exists():
            tqdm.write(f"  Skipping {w_from} → {w_to} (already on disk)")
            continue

        df = fetch_window(pnode_id, w_from, w_to)
        if df.empty:
            tqdm.write(f"  Warning: no data for {w_from} → {w_to}")
            continue

        df = _normalise(df)
        df.to_csv(raw_file, index=False)
        tqdm.write(f"  Saved {len(df):,} rows → {raw_file.name}")

    # ── Merge all raw windows ──────────────────────────────────────────────────
    raw_files = sorted(raw_node_dir.glob("*.csv"))
    if not raw_files:
        print("No data downloaded.")
        return

    print(f"\nMerging {len(raw_files)} window file(s)...")
    result = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True)
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    result = (result
              .drop_duplicates(subset=["timestamp_utc"])
              .sort_values("timestamp_utc")
              .reset_index(drop=True))

    first = _mmyy(result["timestamp_utc"].iloc[0])
    last  = _mmyy(result["timestamp_utc"].iloc[-1])
    output_file = OUTPUT_DIR / f"day_ahead_pjm_{pnode_id}_{first}_{last}.csv"

    result.to_csv(output_file, index=False)
    print(f"Saved {len(result):,} rows to {output_file}")
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
