import pandas as pd
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

SPAIN_PREFIX = "marginalpdbc_"
PORTUGAL_PREFIX = "marginalpdbcpt_"

SPAIN_MASTER = OUTPUT_DIR / "day_ahead_spain.csv"
PORTUGAL_MASTER = OUTPUT_DIR / "day_ahead_portugal.csv"

# -----------------------------
# Helper functions
# -----------------------------
def parse_file(file_path):
    """
    Parse a raw file (.1/.2/.3/.4) into long format: YYYYMMDD, HH:MM, price.
    Handles both hourly (periods 1-24/25) and 15-minute (periods 1-96+) formats.
    The format is detected per-file by counting data rows.
    """
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data_lines = [l.strip() for l in lines[1:] if l.strip() and len(l.strip().split(";")) >= 5]
    is_quarterly = len(data_lines) > 25  # DST hourly days have at most 25 rows
    for line in data_lines:
        parts = line.split(";")
        year, month, day, period_str, price = parts[0], parts[1], parts[2], parts[3], parts[4]
        yyyymmdd = f"{year}{month.zfill(2)}{day.zfill(2)}"
        period = int(period_str)
        if is_quarterly:
            total_minutes = period * 15
            hhmm = f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"
        else:
            hhmm = f"{period:02d}:00"
        try:
            price_val = float(price.replace(",", "."))
        except ValueError:
            price_val = None
        rows.append([yyyymmdd, hhmm, price_val])
    return rows

def process_files(prefix, output_file, market_name):
    # Find all files with the given prefix
    files = sorted(f for ext in (".1", ".2", ".3", ".4") for f in RAW_DIR.glob(f"{prefix}*{ext}"))
    all_rows = []
    for file_path in tqdm(files, desc=f"Processing {market_name} files"):
        all_rows.extend(parse_file(file_path))
    # Create DataFrame
    df = pd.DataFrame(all_rows, columns=["YYYYMMDD", "HH:MM", "price"])
    # Sort by date and hour, then drop exact duplicates
    df = df.sort_values(["YYYYMMDD", "HH:MM"]).drop_duplicates(subset=["YYYYMMDD", "HH:MM"])
    # Save CSV
    df.to_csv(output_file, index=False)

    # -----------------------------
    # Summary
    # -----------------------------
    if not df.empty:
        first_date = df["YYYYMMDD"].iloc[0]
        last_date = df["YYYYMMDD"].iloc[-1]
        total_rows = len(df)
        print(f"\n{market_name} Summary:")
        print(f"First available date: {first_date}")
        print(f"Last available date: {last_date}")
        print(f"Total price records: {total_rows}\n")
    else:
        print(f"\n{market_name} Summary: No data found.\n")

# -----------------------------
# Main
# -----------------------------

# -----------------------------
# Multivariate dataset creation
# -----------------------------
def create_multivariate_dataset(price_csv, output_csv, country_name):
    """
    Create multivariate dataset with:
    price, is_weekend_holiday, hour, day_of_week, month, is_dst, + weather features.
    All timestamps aligned in UTC.
    """
    df = pd.read_csv(price_csv)

    # Ensure string types before concatenation (robust to int columns)
    df["YYYYMMDD"] = df["YYYYMMDD"].astype(str)
    df["HH:MM"] = df["HH:MM"].astype(str).str.zfill(5)

    # Extract year from YYYYMMDD
    df["year"] = df["YYYYMMDD"].str[:4].astype(int)

    df["timestamp"] = pd.to_datetime(
        df["YYYYMMDD"] + " " + df["HH:MM"],
        format="%Y%m%d %H:%M",
        errors="coerce"
    )

    # Drop rows where timestamp failed to parse
    df = df.dropna(subset=["timestamp"])

    # Handle DST edge cases:
    # - nonexistent times (spring forward) → shift forward
    # - ambiguous times (fall back) → drop ambiguous rows
    df = df.sort_values("timestamp")  # ensure chronological
    df["timestamp"] = df["timestamp"].dt.tz_localize(
        "Europe/Madrid",
        nonexistent="shift_forward",
        ambiguous="NaT"
    )

    # Drop any rows that became NaT due to ambiguous fall-back
    df = df.dropna(subset=["timestamp"])

    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month

    # Safe DST calculation
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["is_dst"] = df["timestamp"].apply(lambda x: bool(x.dst()))

    # Weekend flag
    df["is_weekend_holiday"] = df["day_of_week"] >= 5

    # -----------------------------
    # Merge weather data
    # -----------------------------
    weather_files = list(Path("data").glob("weather_*.csv"))

    for wf in weather_files:
        city_name = wf.stem.replace("weather_", "")
        wdf = pd.read_csv(wf)

        if "timestamp" not in wdf.columns:
            continue

        wdf["timestamp"] = pd.to_datetime(wdf["timestamp"], utc=True)

        # Prefix weather columns
        weather_cols = [c for c in wdf.columns if c != "timestamp"]
        wdf = wdf.rename(columns={c: f"{city_name}_{c}" for c in weather_cols})

        # Merge on UTC timestamp
        df = df.merge(wdf, on="timestamp", how="left")

    # Keep timestamp for sorting but do not include in final features yet
    base_cols = ["year","month", "hour", "minute", "price", "is_weekend_holiday", "day_of_week", "is_dst"]
    all_cols = ["timestamp"] + base_cols
    weather_cols = [c for c in df.columns if c not in ["YYYYMMDD", "HH:MM", "timestamp"] + base_cols]
    all_cols.extend(weather_cols)

    df = df[all_cols]

    # Sort by timestamp
    df = df.sort_values("timestamp")

    # Drop timestamp column after sorting
    df = df.drop(columns=["timestamp"])

    # Save
    df.to_csv(output_csv, index=False)

    print(f"{country_name} multivariate dataset saved to {output_csv}")


def main():
    process_files(SPAIN_PREFIX, SPAIN_MASTER, "Spain")
    process_files(PORTUGAL_PREFIX, PORTUGAL_MASTER, "Portugal")

    # Create multivariate datasets
    create_multivariate_dataset(SPAIN_MASTER, OUTPUT_DIR / "day_ahead_spain_multivariate.csv", "Spain")
    create_multivariate_dataset(PORTUGAL_MASTER, OUTPUT_DIR / "day_ahead_portugal_multivariate.csv", "Portugal")

    print("Processing complete for both Spain and Portugal.")

if __name__ == "__main__":
    main()