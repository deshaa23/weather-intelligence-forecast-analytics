from datetime import datetime, timezone

import pytest

from src.transformation.clean_weather import hourly_json_to_dataframe


def sample_payload():
    return {
        "hourly": {
            "time": ["2026-10-17T10:00", "2026-10-17T11:00"],
            "temperature_2m": [29.1, 30.0],
            "relative_humidity_2m": [65, 61],
            "precipitation_probability": [20, 10],
            "precipitation": [0.0, 0.0],
            "wind_speed_10m": [8.0, 9.0],
            "weather_code": [2, 1],
        }
    }


def test_transform_returns_rows():
    df = hourly_json_to_dataframe(
        sample_payload(), "Pune", datetime.now(timezone.utc)
    )
    assert len(df) == 2
    assert df.iloc[0]["temperature_forecast_c"] == 29.1


def test_invalid_humidity_raises():
    payload = sample_payload()
    payload["hourly"]["relative_humidity_2m"][0] = 120
    with pytest.raises(ValueError):
        hourly_json_to_dataframe(
            payload, "Pune", datetime.now(timezone.utc)
        )
