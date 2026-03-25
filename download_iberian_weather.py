from datetime import datetime
import meteostat as ms
import pandas as pd

ms.config.block_large_requests = False

START = datetime(2018, 1, 1)
END = datetime.now().replace(minute=0, second=0, microsecond=0)

# Madrid-Barajas (40.47°N, 3.57°W, 582m)
madrid = "08221"
# Lisbon Humberto Delgado (38.77°N, 9.13°W, 113m)
lisbon = "08536"

for name, station_id, out_path in [
    ("Madrid", madrid, "data/weather_madrid.csv"),
    ("Lisbon", lisbon, "data/weather_lisbon.csv"),
]:
    # stations = ms.stations.nearby(50.05, 8.68)
    # station = stations.fetch(1)
    # print(station)
    print(f"Fetching hourly weather for {name} with ID {station_id}...")
    data = ms.hourly(ms.Station(id=station_id), START, END)
    df = data.fetch()
    print(df)
    df.index.name = "time"
    df.to_csv(out_path)
    print(f"  Saved {len(df):,} rows to {out_path}")
