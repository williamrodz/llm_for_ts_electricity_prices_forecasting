import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Maps area code (lowercase, as used in filenames) → representative weather city
AREA_TO_CITY = {
    "no_1":  "oslo",
    "no_2":  "kristiansand",
    "no_3":  "trondheim",
    "no_4":  "tromso",
    "no_5":  "bergen",
    "se_1":  "lulea",
    "se_2":  "sundsvall",
    "se_3":  "stockholm",
    "se_4":  "malmo",
    "dk_1":  "aarhus",
    "dk_2":  "copenhagen",
    "fi":    "helsinki",
    "ee":    "tallinn",
    "lv":    "riga",
    "lt":    "vilnius",
    "de_lu": "berlin",
    "fr":    "paris",
    "nl":    "amsterdam",
    "be":    "brussels",
    "at":    "vienna",
    "gb":    "london",
}

# Local timezone for time-of-day features (demand patterns are local, not UTC)
AREA_TO_TZ = {
    "no_1":  "Europe/Oslo",
    "no_2":  "Europe/Oslo",
    "no_3":  "Europe/Oslo",
    "no_4":  "Europe/Oslo",
    "no_5":  "Europe/Oslo",
    "se_1":  "Europe/Stockholm",
    "se_2":  "Europe/Stockholm",
    "se_3":  "Europe/Stockholm",
    "se_4":  "Europe/Stockholm",
    "dk_1":  "Europe/Copenhagen",
    "dk_2":  "Europe/Copenhagen",
    "fi":    "Europe/Helsinki",
    "ee":    "Europe/Tallinn",
    "lv":    "Europe/Riga",
    "lt":    "Europe/Vilnius",
    "de_lu": "Europe/Berlin",
    "fr":    "Europe/Paris",
    "nl":    "Europe/Amsterdam",
    "be":    "Europe/Brussels",
    "at":    "Europe/Vienna",
    "gb":    "Europe/London",
}


# -----------------------------
# Helper functions
# -----------------------------
def load_weather(city: str) -> pd.DataFrame | None:
    """
    Load a Meteostat weather CSV and return it with a UTC timestamp_utc column.
    Returns None if the file doesn't exist.
    Meteostat saves the DatetimeIndex as 'time' (UTC naive); we parse it as UTC.
    """
    wf = DATA_DIR / f"weather_{city}.csv"
    if not wf.exists():
        print(f"  Warning: weather file not found for '{city}' ({wf}) — skipping weather merge.")
        return None

    wdf = pd.read_csv(wf)
    wdf = wdf.rename(columns={"time": "timestamp_utc"})
    wdf["timestamp_utc"] = pd.to_datetime(wdf["timestamp_utc"], utc=True)

    weather_cols = [c for c in wdf.columns if c != "timestamp_utc"]
    wdf = wdf.rename(columns={c: f"{city}_{c}" for c in weather_cols})
    return wdf


def process_area(price_csv: Path, area: str) -> None:
    """
    Build a multivariate dataset for one Nord Pool bidding zone.

    Reads the price CSV, adds local time features, merges the corresponding
    city's weather data, and saves to data/day_ahead_{area}_multivariate_{MMYY}_{MMYY}.csv.
    """
    city = AREA_TO_CITY.get(area)
    tz   = AREA_TO_TZ.get(area, "UTC")

    df = pd.read_csv(price_csv)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Elexon GB file uses "price" (GBP); rename for clarity
    if area == "gb" and "price" in df.columns and "price_gbp_mwh" not in df.columns:
        df = df.rename(columns={"price": "price_gbp_mwh"})

    # ── Local time features ────────────────────────────────────────────────────
    local = df["timestamp_utc"].dt.tz_convert(tz)

    df["year"]               = local.dt.year
    df["month"]              = local.dt.month
    df["day_of_month"]       = local.dt.day
    df["hour"]               = local.dt.hour
    df["day_of_week"]        = local.dt.weekday          # 0 = Monday
    df["is_weekend_holiday"] = local.dt.weekday >= 5
    df["is_dst"]             = local.apply(lambda x: bool(x.dst()))

    # ── Weather merge ──────────────────────────────────────────────────────────
    if city:
        wdf = load_weather(city)
        if wdf is not None:
            df = df.merge(wdf, on="timestamp_utc", how="left")

    # ── Column ordering ────────────────────────────────────────────────────────
    time_cols  = ["timestamp_utc", "year", "month", "day_of_month", "hour",
                  "day_of_week", "is_weekend_holiday", "is_dst"]
    price_col  = [c for c in df.columns if "price" in c]
    other_cols = [c for c in df.columns if c not in time_cols + price_col]
    df = df[time_cols + price_col + other_cols]

    # ── Save ───────────────────────────────────────────────────────────────────
    first = df["timestamp_utc"].iloc[0].strftime("%m%y")
    last  = df["timestamp_utc"].iloc[-1].strftime("%m%y")
    output_file = OUTPUT_DIR / f"day_ahead_{area}_multivariate_{first}_{last}.csv"

    df.to_csv(output_file, index=False)
    print(f"  Saved {len(df):,} rows → {output_file.name}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build multivariate datasets from downloaded Nord Pool price CSVs."
    )
    parser.add_argument(
        "--areas", nargs="+", default=None, metavar="AREA",
        help="Area codes to process (default: all found in data/). E.g. --areas fi se_3 no_2",
    )
    args = parser.parse_args()

    # Discover all downloaded Nord Pool price files
    price_files = sorted(DATA_DIR.glob("day_ahead_nordpool_*.csv"))

    # Extract area from filename: day_ahead_nordpool_{area}_{MMYY}_{MMYY}.csv
    pattern = re.compile(r"day_ahead_nordpool_(.+)_\d{4}_\d{4}\.csv")
    found = []
    for pf in price_files:
        m = pattern.match(pf.name)
        if m:
            found.append((m.group(1), pf))

    # GB override: the ENTSO-E GB file ends Dec 2020 (Brexit).
    # Use the Elexon file (day_ahead_gb_APXMIDP_*.csv) for full coverage instead.
    elexon_files = sorted(DATA_DIR.glob("day_ahead_gb_APXMIDP_*.csv"))
    if elexon_files:
        found = [(area, pf) for area, pf in found if area != "gb"]
        found.append(("gb", elexon_files[-1]))

    if not found:
        print("No price files found in data/. Run download_nordpool_prices.py first.")
        return

    if args.areas:
        requested = {a.lower() for a in args.areas}
        found = [(area, pf) for area, pf in found if area in requested]
        missing = requested - {area for area, _ in found}
        if missing:
            print(f"Warning: no downloaded price files found for: {sorted(missing)}")

    if not found:
        print("Nothing to process.")
        return

    print(f"Processing {len(found)} area(s)...\n")
    for area, pf in tqdm(found, desc="Areas", unit=" area"):
        city = AREA_TO_CITY.get(area, "—")
        tz   = AREA_TO_TZ.get(area, "UTC")
        tqdm.write(f"  {area:8s} | city: {city:<14} | tz: {tz}")
        try:
            process_area(pf, area)
        except Exception as e:
            tqdm.write(f"  ERROR processing {area}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
