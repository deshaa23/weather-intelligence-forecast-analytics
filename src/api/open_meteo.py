"""Reliable client for the Open-Meteo forecast API."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_hourly_forecast(
    latitude: float,
    longitude: float,
    city_name: str,
    timezone_name: str = "Asia/Kolkata",
    forecast_days: int = 7,
    max_attempts: int = 4,
) -> tuple[dict[str, Any], datetime]:
    """Fetch hourly forecast JSON with retries and response validation."""

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation_probability",
            "precipitation",
            "wind_speed_10m",
            "weather_code",
        ],
        "forecast_days": forecast_days,
        "timezone": timezone_name,
    }

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                FORECAST_URL,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()

            if "hourly" not in payload or "time" not in payload["hourly"]:
                raise ValueError(
                    "API response is missing expected hourly forecast data."
                )

            payload["_project_metadata"] = {
                "city_name": city_name,
                "provider": "open-meteo",
            }

            return payload, datetime.now(timezone.utc)

        except (requests.Timeout, requests.ConnectionError) as exc:
            error = exc

        except requests.HTTPError as exc:
            status_code = exc.response.status_code

            # Retry temporary server errors and rate limiting.
            if status_code not in (429, 500, 502, 503, 504):
                raise

            error = exc

        if attempt < max_attempts:
            wait_seconds = 2 ** attempt
            print(
                f"{city_name}: API request failed on attempt "
                f"{attempt}/{max_attempts}. "
                f"Retrying in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)
        else:
            raise RuntimeError(
                f"Failed to fetch weather for {city_name} after "
                f"{max_attempts} attempts. Last error: {error}"
            ) from error

    raise RuntimeError(f"Weather retrieval failed for {city_name}.")