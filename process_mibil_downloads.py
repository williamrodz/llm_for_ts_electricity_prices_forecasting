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
    Parse a raw file (.1/.2/.3) into long format: YYYYMMDD, HH:MM, price
    """
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Skip the first line (header)
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(";")
        if len(parts) < 5:
            continue  # skip malformed lines
        year, month, day, hour, price = parts[0], parts[1], parts[2], parts[3], parts[4]
        yyyymmdd = f"{year}{month.zfill(2)}{day.zfill(2)}"
        hhmm = f"{hour.zfill(2)}:00"
        try:
            price_val = float(price.replace(",", "."))  # handle comma decimal if present
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
        print(f"Total hourly records: {total_rows}\n")
    else:
        print(f"\n{market_name} Summary: No data found.\n")

# -----------------------------
# Main
# -----------------------------
def main():
    process_files(SPAIN_PREFIX, SPAIN_MASTER, "Spain")
    process_files(PORTUGAL_PREFIX, PORTUGAL_MASTER, "Portugal")
    print("Processing complete for both Spain and Portugal.")

if __name__ == "__main__":
    main()