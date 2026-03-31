import argparse
from datetime import datetime
from pathlib import Path

import meteostat as ms
import pandas as pd

ms.config.block_large_requests = False

# -----------------------------
# Config
# -----------------------------
START     = datetime(2015, 1, 1)
END       = datetime.now().replace(minute=0, second=0, microsecond=0)
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# WMO station IDs and coordinates.
# Organised by Nord Pool bidding zone, then other markets.
# Each entry: "city_key": (station_id, "Description (lat, lon, elev)")
STATIONS = {
    # ── Norway ────────────────────────────────────────────────────────────────
    # NO1 — Oslo / Southeast
    "oslo":         ("01492", "Oslo Blindern          (59.94°N,  10.72°E,  94m)"),
    # NO2 — Southwest
    "kristiansand": ("01452", "Kristiansand Kjevik    (58.20°N,   8.08°E,  12m)"),
    # NO3 — Central
    "trondheim":    ("01488", "Trondheim Vaernes      (63.46°N,  10.92°E,  14m)"),
    # NO4 — Northern
    "tromso":       ("01025", "Tromsø Airport         (69.68°N,  18.92°E,  10m)"),
    # NO5 — West
    "bergen":       ("01317", "Bergen Florida         (60.38°N,   5.33°E,  12m)"),

    # ── Sweden ────────────────────────────────────────────────────────────────
    # SE1 — Northern
    "lulea":        ("02212", "Luleå Airport          (65.54°N,  22.12°E,  14m)"),
    # SE2 — North-Central
    "sundsvall":    ("02261", "Sundsvall-Härnösand    (62.53°N,  17.44°E,  14m)"),
    # SE3 — South-Central (largest zone)
    "stockholm":    ("02443", "Stockholm Arlanda      (59.65°N,  17.95°E,  61m)"),
    # SE4 — Southern
    "malmo":        ("02600", "Malmö Airport          (55.54°N,  13.37°E,  21m)"),

    # ── Denmark ───────────────────────────────────────────────────────────────
    # DK1 — Jutland & Funen / West
    "aarhus":       ("06073", "Aarhus (Tirstrup)      (56.30°N,  10.62°E,  57m)"),
    # DK2 — Zealand / East
    "copenhagen":   ("06180", "Copenhagen Kastrup     (55.63°N,  12.65°E,   5m)"),

    # ── Finland ───────────────────────────────────────────────────────────────
    "helsinki":     ("02974", "Helsinki Vantaa        (60.32°N,  24.97°E,  47m)"),

    # ── Baltic states ─────────────────────────────────────────────────────────
    "tallinn":      ("26038", "Tallinn Airport        (59.41°N,  24.80°E,  40m)"),
    "riga":         ("26422", "Riga Airport           (56.92°N,  23.97°E,  10m)"),
    "vilnius":      ("26730", "Vilnius Airport        (54.63°N,  25.28°E, 162m)"),

    # ── Continental Europe (ENTSO-E single zones) ────────────────────────────
    # DE_LU — Germany/Luxembourg
    "berlin":       ("10382", "Berlin-Tegel           (52.56°N,  13.32°E,  36m)"),
    # FR — France
    "paris":        ("07157", "Paris Charles de Gaulle(49.02°N,   2.53°E, 109m)"),
    # NL — Netherlands
    "amsterdam":    ("06240", "Amsterdam Schiphol     (52.30°N,   4.77°E,  -3m)"),
    # BE — Belgium
    "brussels":     ("06451", "Brussels Zaventem      (50.90°N,   4.53°E,  58m)"),
    # AT — Austria
    "vienna":       ("11036", "Vienna Schwechat       (48.11°N,  16.57°E, 183m)"),

    # ── Great Britain ─────────────────────────────────────────────────────────
    "london":       ("03779", "London Heathrow        (51.48°N,   0.45°W,  25m)"),

    # ── Iberian Peninsula (OMIE) ──────────────────────────────────────────────
    "madrid":       ("08221", "Madrid-Barajas         (40.47°N,   3.57°W, 582m)"),
    "lisbon":       ("08536", "Lisbon Humberto Delgado(38.77°N,   9.13°W, 113m)"),
}


# -----------------------------
# Download
# -----------------------------
def download_cities(city_keys: list[str]) -> None:
    for key in city_keys:
        if key not in STATIONS:
            print(f"Unknown city '{key}' — skipping. Valid keys: {list(STATIONS)}")
            continue

        station_id, description = STATIONS[key]
        out_path = OUTPUT_DIR / f"weather_{key}.csv"

        print(f"Fetching hourly weather: {key} | {description} | station {station_id}")
        data = ms.hourly(ms.Station(id=station_id), START, END)
        df = data.fetch()

        if df.empty:
            print(f"  Warning: no data returned for {key} (station {station_id})")
            continue

        df.index.name = "time"
        df.to_csv(out_path)
        print(f"  Saved {len(df):,} rows to {out_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    global START, END
    all_keys = list(STATIONS.keys())

    parser = argparse.ArgumentParser(
        description="Download hourly weather data for Nord Pool bidding zone cities via Meteostat.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available cities:\n" + "\n".join(
            f"  {k:<14} {v}" for k, v in STATIONS.items()
        ),
    )
    parser.add_argument(
        "--cities", nargs="+", default=all_keys, metavar="CITY",
        help="City keys to download (default: all). E.g. --cities oslo stockholm helsinki",
    )
    parser.add_argument(
        "--start", type=str, default=START.strftime("%Y-%m-%d"),
        help=f"Start date YYYY-MM-DD (default: {START.strftime('%Y-%m-%d')})",
    )
    parser.add_argument(
        "--end", type=str, default=END.strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args()

    START = datetime.strptime(args.start, "%Y-%m-%d")
    END   = datetime.strptime(args.end,   "%Y-%m-%d")

    download_cities([c.lower() for c in args.cities])


if __name__ == "__main__":
    main()
