from pathlib import Path
import time

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical_weather_reanalysis.csv"
)

API_URL = "https://archive-api.open-meteo.com/v1/archive"

CITIES = {
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
}


def fetch_city_weather(city_name, latitude, longitude):
    """Collect hourly historical weather for one city."""

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2024-01-01",
        "end_date": "2025-12-31",
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m,"
            "weather_code"
        ),
        "timezone": "Asia/Kolkata",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        # ERA5 supports all the requested variables.
        "models": "era5",
    }

    for attempt in range(1, 5):
        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=120,
            )
            response.raise_for_status()

            payload = response.json()

            if payload.get("error"):
                raise RuntimeError(
                    payload.get("reason", "API returned an error")
                )

            hourly = payload["hourly"]

            df = pd.DataFrame({
                "timestamp_local": hourly["time"],
                "temperature_2m_c": hourly["temperature_2m"],
                "relative_humidity_2m_pct": (
                    hourly["relative_humidity_2m"]
                ),
                "precipitation_mm": hourly["precipitation"],
                "wind_speed_10m_kmh": hourly["wind_speed_10m"],
                "weather_code": hourly["weather_code"],
            })

            df.insert(0, "city_name", city_name)

            df["timestamp_local"] = pd.to_datetime(
                df["timestamp_local"],
                errors="raise",
            )

            return df

        except (requests.RequestException, ValueError, KeyError) as error:
            if attempt == 4:
                raise RuntimeError(
                    f"Failed to collect data for {city_name}"
                ) from error

            wait_seconds = 2 ** attempt
            print(f"Attempt {attempt} failed: {error}")
            print(f"Retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)

    raise RuntimeError(f"Could not collect data for {city_name}")


def main():
    """Collect, combine, validate, and save the dataset."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    city_data = []

    for city_name, (latitude, longitude) in CITIES.items():
        print(f"\nFetching historical weather for {city_name}...")

        df = fetch_city_weather(
            city_name,
            latitude,
            longitude,
        )

        print(f"Records collected: {len(df):,}")
        city_data.append(df)

        # Brief pause between API requests.
        time.sleep(1)

    combined_df = pd.concat(city_data, ignore_index=True)

    combined_df = combined_df.sort_values(
        ["city_name", "timestamp_local"]
    ).reset_index(drop=True)

    duplicate_count = combined_df.duplicated(
        ["city_name", "timestamp_local"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate city/hour records."
        )

    print("\nMissing values:")
    print(combined_df.isna().sum().to_string())

    combined_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 50)
    print("HISTORICAL WEATHER COLLECTION COMPLETED")
    print("=" * 50)
    print(f"Total records: {len(combined_df):,}")
    print(f"Number of cities: {combined_df['city_name'].nunique()}")
    print(f"Output file: {OUTPUT_PATH}")
    print("\nRecords per city:")
    print(combined_df.groupby("city_name").size().to_string())


if __name__ == "__main__":
    main()