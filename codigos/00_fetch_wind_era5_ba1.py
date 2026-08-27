from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
OCEAN_CSV = BASE_DIR / "dadosSimcosta" / "SIMCOSTA_BA-1_OCEAN_2019-08-02_2026-07-25.csv"
OUT_DIR = BASE_DIR / "dadosSimcosta"

# BA-1 buoy position, from the OCEAN file header (/latitude, /longitude).
LATITUDE = -12.9895333
LONGITUDE = -38.5418

API_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "wind_speed_10m,wind_direction_10m,wind_gusts_10m"


def read_ocean_date_range() -> tuple[str, str]:
    df = pd.read_csv(
        OCEAN_CSV,
        comment="/",
        na_values=["NULL", "null", "", "NaN"],
        usecols=["YEAR", "MONTH", "DAY"],
    )
    dt = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")
    return dt.min().strftime("%Y-%m-%d"), dt.max().strftime("%Y-%m-%d")


def fetch_era5_wind(start_date: str, end_date: str) -> tuple[pd.DataFrame, float, float]:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_VARS,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    response = requests.get(API_URL, params=params, timeout=180)
    response.raise_for_status()
    payload = response.json()
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["datetime_utc"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop(columns=["time"]).set_index("datetime_utc")
    return df, payload["latitude"], payload["longitude"]


def main() -> None:
    start_date, end_date = read_ocean_date_range()
    wind, grid_lat, grid_lon = fetch_era5_wind(start_date, end_date)

    out_path = OUT_DIR / f"ERA5_BA-1_WIND_{start_date}_{end_date}.csv"
    header_lines = [
        "/ERA5 reanalysis wind at the nearest grid point to the SiMCosta BA-1 buoy",
        f"/buoy_latitude = {LATITUDE}",
        f"/buoy_longitude = {LONGITUDE}",
        f"/era5_grid_latitude = {grid_lat}",
        f"/era5_grid_longitude = {grid_lon}",
        "/source = Open-Meteo Historical Weather API (ERA5 reanalysis), https://open-meteo.com",
        "/IMPORTANT: this is NOT in-situ buoy-measured wind. It is a reanalysis model",
        "/estimate at the nearest ocean grid cell (~0.25 deg resolution), sampled at 10 m",
        "/height (the SiMCosta anemometer, where present, measures at 3 m). Use as a",
        "/physically independent covariate proxy for wind forcing at BA-1, not as ground",
        "/truth wind measured at the buoy itself.",
        "/wind_speed_10m: 10 m wind speed (m/s)",
        "/wind_direction_10m: 10 m wind direction, meteorological convention - degrees the wind blows FROM (deg)",
        "/wind_gusts_10m: 10 m wind gusts (m/s)",
        "/Missing = NULL",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write("\n".join(header_lines) + "\n")
        wind.to_csv(f, na_rep="NULL")

    print(f"Wind data saved to {out_path} ({len(wind)} hourly records, {start_date} to {end_date})")
    print(f"Buoy coords: ({LATITUDE}, {LONGITUDE}) -> nearest ERA5 grid cell: ({grid_lat}, {grid_lon})")


if __name__ == "__main__":
    main()
