"""Transform the nested hourly API response into a tidy DataFrame."""
from __future__ import annotations

import pandas as pd


def hourly_json_to_dataframe(
    payload: dict,
    city_name: str,
    forecast_issued_at,
) -> pd.DataFrame:
    hourly = payload["hourly"]
    required = [
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation_probability",
        "precipitation",
        "wind_speed_10m",
        "weather_code",
    ]
    missing = [column for column in required if column not in hourly]
    if missing:
        raise ValueError(f"Missing expected API fields: {missing}")

    df = pd.DataFrame({
        "city_name": city_name,
        "provider": "open-meteo",
        "forecast_issued_at": forecast_issued_at,
        "forecast_for": pd.to_datetime(hourly["time"]),
        "temperature_forecast_c": hourly["temperature_2m"],
        "humidity_forecast_pct": hourly["relative_humidity_2m"],
        "precipitation_probability_pct": hourly["precipitation_probability"],
        "precipitation_forecast_mm": hourly["precipitation"],
        "wind_speed_forecast_kmh": hourly["wind_speed_10m"],
        "weather_code": hourly["weather_code"],
    })

    # Basic range validation: missing values are allowed, invalid non-null values are not.
    if not df["humidity_forecast_pct"].dropna().between(0, 100).all():
        raise ValueError("Humidity values fall outside the expected 0–100% range.")
    if not df["precipitation_probability_pct"].dropna().between(0, 100).all():
        raise ValueError("Precipitation probability falls outside 0–100%.")
    if (df["precipitation_forecast_mm"].dropna() < 0).any():
        raise ValueError("Precipitation cannot be negative.")
    if (df["wind_speed_forecast_kmh"].dropna() < 0).any():
        raise ValueError("Wind speed cannot be negative.")

    return df
