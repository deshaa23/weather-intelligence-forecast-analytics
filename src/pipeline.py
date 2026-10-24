"""Phase 1 pipeline: fetch, transform, and save weather data for multiple cities."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.api.open_meteo import fetch_hourly_forecast
from src.transformation.clean_weather import hourly_json_to_dataframe

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

CITIES = [
    {
        "city_name": "Pune",
        "latitude": 18.5204,
        "longitude": 73.8567,
        "timezone": "Asia/Kolkata",
    },
    {
        "city_name": "Mumbai",
        "latitude": 19.0760,
        "longitude": 72.8777,
        "timezone": "Asia/Kolkata",
    },
    {
        "city_name": "Delhi",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "timezone": "Asia/Kolkata",
    },
    {
        "city_name": "Bengaluru",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "timezone": "Asia/Kolkata",
    },
    {
        "city_name": "Chennai",
        "latitude": 13.0827,
        "longitude": 80.2707,
        "timezone": "Asia/Kolkata",
    },
    {
        "city_name": "Hyderabad",
        "latitude": 17.3850,
        "longitude": 78.4867,
        "timezone": "Asia/Kolkata",
    },
    {
        "city_name": "Kolkata",
        "latitude": 22.5726,
        "longitude": 88.3639,
        "timezone": "Asia/Kolkata",
    },
]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    city_dataframes = []

    for city in CITIES:
        print(f"\nFetching forecast for {city['city_name']}...")

        payload, retrieved_at = fetch_hourly_forecast(
            latitude=city["latitude"],
            longitude=city["longitude"],
            city_name=city["city_name"],
            timezone_name=city["timezone"],
            forecast_days=7,
        )

        city_slug = city["city_name"].lower()

        raw_path = (
            RAW_DIR
            / f"{city_slug}_open_meteo_{retrieved_at:%Y%m%dT%H%M%SZ}.json"
        )
        raw_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        df = hourly_json_to_dataframe(
            payload=payload,
            city_name=city["city_name"],
            forecast_issued_at=retrieved_at,
        )

        city_dataframes.append(df)

        print(f"  Forecast records: {len(df)}")
        print(f"  Raw JSON saved: {raw_path}")

    combined_df = pd.concat(city_dataframes, ignore_index=True)

    csv_path = PROCESSED_DIR / "india_weather_forecast.csv"
    combined_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 55)
    print("Multi-city weather pipeline completed successfully.")
    print(f"Cities processed: {combined_df['city_name'].nunique()}")
    print(f"Total forecast records: {len(combined_df)}")
    print(f"Processed CSV: {csv_path}")
    print("\nRecords per city:")
    print(combined_df.groupby("city_name").size().to_string())
    print("\nPreview:")
    print(combined_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()