#!/usr/bin/env python3
"""
download_pjm_weather.py
Fetch hourly weather data for a PJM pricing node via meteostat.

Steps:
  1. Query the PJM API for the node name (pnode_id → pnode_name)
  2. Resolve lat/lon from a static zone table, or fall back to Nominatim geocoding
  3. Find the nearest meteostat weather station
  4. Download hourly data and save to data/weather_pjm_{pnode_id}.csv

Usage:
  python download_pjm_weather.py --node 51291
  python download_pjm_weather.py --node 51291 --start 2020-01-01 --end 2023-12-31
  python download_pjm_weather.py --node 51291 --station 72408  # override station
"""

import argparse
import os
import time
from datetime import datetime

import meteostat as ms
import requests
from dotenv import load_dotenv

load_dotenv()

ms.config.block_large_requests = False

# ── Config ─────────────────────────────────────────────────────────────────────
PJM_API_URL = "https://api.pjm.com/api/v1/pnode"
API_KEY     = os.environ.get("PJM_API_KEY", "")
DATA_DIR    = "data"

# Approximate centre coordinates for PJM's 21 load zones.
# Indexed by the pnode_name string returned by the PJM API.
ZONE_COORDS: dict[str, tuple[float, float, str]] = {
    "AECO":    (39.36, -74.42, "Atlantic City, NJ"),   # 51291
    "AEP":     (39.96, -82.99, "Columbus, OH"),        # 8445784
    "APS":     (38.35, -81.63, "Charleston, WV"),      # 8394954
    "ATSI":    (41.40, -81.85, "Cleveland, OH"),       # 116013753
    "BGE":     (39.29, -76.61, "Baltimore, MD"),       # 51292
    "COMED":   (41.88, -87.63, "Chicago, IL"),         # 33092371
    "DAY":     (39.76, -84.19, "Dayton, OH"),          # 34508503
    "DEOK":    (39.10, -84.51, "Cincinnati, OH"),      # 124076095
    "DOM":     (37.54, -77.43, "Richmond, VA"),        # 34964545
    "DPL":     (39.16, -75.52, "Dover, DE"),           # 51293
    "DUQ":     (40.44, -79.99, "Pittsburgh, PA"),      # 37737283
    "EKPC":    (37.99, -84.18, "Winchester, KY"),      # 970242670
    "JCPL":    (40.80, -74.48, "Morristown, NJ"),      # 51295
    "METED":   (40.34, -75.93, "Reading, PA"),         # 51296
    "PECO":    (39.95, -75.16, "Philadelphia, PA"),    # 51297
    "PENELEC": (40.79, -77.86, "State College, PA"),   # 51300
    "PEPCO":   (38.91, -77.04, "Washington, DC"),      # 51298
    "PPL":     (40.60, -75.49, "Allentown, PA"),       # 51299
    "PSEG":    (40.74, -74.17, "Newark, NJ"),          # 51301
    "RECO":    (41.11, -74.04, "Spring Valley, NY"),   # 7633629
    "PJM-RTO": (40.09, -75.38, "Valley Forge, PA"),   # 1
}


# ── Node resolution ────────────────────────────────────────────────────────────
def get_node_name(pnode_id: int) -> str:
    resp = requests.get(
        PJM_API_URL,
        params={"pnode_id": pnode_id, "rowCount": 1, "startRow": 1},
        headers={"Ocp-Apim-Subscription-Key": API_KEY},
        timeout=15,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        raise ValueError(f"Node {pnode_id} not found in PJM API.")
    return items[0]["pnode_name"]


def resolve_coords(node_name: str) -> tuple[float, float, str]:
    """Return (lat, lon, description) for a PJM node name.

    Checks the static zone table first, then falls back to Nominatim geocoding
    for named bus nodes not in the table.
    """
    # Exact match
    if node_name in ZONE_COORDS:
        return ZONE_COORDS[node_name]

    # Prefix match — e.g. "AECO 345KV" → AECO entry
    upper = node_name.upper()
    for zone, coords in ZONE_COORDS.items():
        if upper.startswith(zone.upper()):
            print(f"  Matched '{node_name}' to zone '{zone}' by prefix.")
            return coords

    # Nominatim fallback for physical bus nodes
    print(f"  Node '{node_name}' not in zone table — trying Nominatim geocoding...")
    time.sleep(1)  # respect Nominatim's 1 req/s rate limit
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": node_name, "format": "json", "limit": 1},
        headers={"User-Agent": "oracle-dissertation/1.0"},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()
    if results:
        lat = float(results[0]["lat"])
        lon = float(results[0]["lon"])
        display = results[0]["display_name"]
        print(f"  Geocoded '{node_name}' → {display}")
        return lat, lon, display

    raise ValueError(
        f"Cannot resolve coordinates for node '{node_name}'. "
        "Add it to ZONE_COORDS manually, or use --lat/--lon to override."
    )


# ── Weather download ───────────────────────────────────────────────────────────
def download_weather(
    pnode_id: int,
    start: datetime,
    end: datetime,
    override_station: str | None = None,
    override_lat: float | None = None,
    override_lon: float | None = None,
) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = f"{DATA_DIR}/weather_pjm_{pnode_id}.csv"

    if override_station:
        print(f"Using manually specified meteostat station: {override_station}")
        station = ms.Station(id=override_station)
    else:
        if override_lat is not None and override_lon is not None:
            lat, lon, loc_desc = override_lat, override_lon, "user-specified"
        else:
            node_name = get_node_name(pnode_id)
            lat, lon, loc_desc = resolve_coords(node_name)
            print(f"Node {pnode_id} = {node_name}  →  ({lat}, {lon})  [{loc_desc}]")

        point = ms.Point(lat, lon)
        station_df = ms.stations.nearby(point).head(1)
        if station_df.empty:
            raise RuntimeError("No meteostat station found near these coordinates.")
        station_id   = station_df.index[0]
        station_name = station_df.iloc[0]["name"]
        print(f"Nearest meteostat station: {station_id} ({station_name})")
        station = ms.Station(id=station_id)

    print(f"Fetching hourly weather {start.date()} → {end.date()}...")
    df = ms.hourly(station, start, end).fetch()
    if df.empty:
        raise RuntimeError(f"No weather data returned. Try a different station with --station.")

    df.index.name = "time"
    df.to_csv(out_path)
    print(f"Saved {len(df):,} rows to {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download hourly weather for a PJM pricing node via meteostat."
    )
    parser.add_argument("--node",    type=int,   default=51291,       help="PJM pnode ID (default: 51291)")
    parser.add_argument("--start",   default="2018-01-01",            help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default=None,                    help="End date YYYY-MM-DD (default: now)")
    parser.add_argument("--station", default=None,                    help="Override meteostat station ID")
    parser.add_argument("--lat",     type=float, default=None,        help="Override latitude")
    parser.add_argument("--lon",     type=float, default=None,        help="Override longitude")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end   = (
        datetime.strptime(args.end, "%Y-%m-%d")
        if args.end
        else datetime.now().replace(minute=0, second=0, microsecond=0)
    )

    download_weather(args.node, start, end, args.station, args.lat, args.lon)
