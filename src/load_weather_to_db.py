import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values


# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load database credentials
load_dotenv(PROJECT_ROOT / ".env")

# Source CSV
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "india_weather_forecast.csv"


def load_weather_data():
    """Load weather forecast data from CSV into PostgreSQL."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Weather CSV not found: {CSV_PATH}"
        )

    # Read the source data
    df = pd.read_csv(CSV_PATH)

    if df.empty:
        raise ValueError("The weather CSV contains no records.")

    # Expected columns in the PostgreSQL table
    columns = [
        "city_name",
        "provider",
        "forecast_issued_at",
        "forecast_for",
        "temperature_forecast_c",
        "humidity_forecast_pct",
        "precipitation_probability_pct",
        "precipitation_forecast_mm",
        "wind_speed_forecast_kmh",
        "weather_code",
    ]

    missing_columns = set(columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required CSV columns: {missing_columns}"
        )

    # Forecast issue timestamps are already UTC-aware.
    df["forecast_issued_at"] = pd.to_datetime(
    df["forecast_issued_at"],
    utc=True,
    errors="raise"
    )

    # Forecast times from Open-Meteo represent India local time.
    forecast_times = pd.to_datetime(
    df["forecast_for"],
    errors="raise"
    )

    if forecast_times.dt.tz is None:
        forecast_times = forecast_times.dt.tz_localize(
            "Asia/Kolkata"
        )

    df["forecast_for"] = forecast_times.dt.tz_convert("UTC")

        # Convert pandas missing values to Python None
        # so PostgreSQL receives SQL NULL where appropriate.
    records = []

    for row in df[columns].itertuples(index=False, name=None):
        record = []

        for value in row:
            if pd.isna(value):
                record.append(None)
            elif isinstance(value, pd.Timestamp):
                record.append(value.to_pydatetime())
            else:
                record.append(value)

        records.append(tuple(record))

    connection = None

    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            connect_timeout=5,
        )

        insert_query = """
    INSERT INTO weather_forecasts (
        city_name,
        provider,
        forecast_issued_at,
        forecast_for,
        temperature_forecast_c,
        humidity_forecast_pct,
        precipitation_probability_pct,
        precipitation_forecast_mm,
        wind_speed_forecast_kmh,
        weather_code
    )
    VALUES %s
    ON CONFLICT (
        city_name,
        forecast_for,
        forecast_issued_at
    )
    DO NOTHING;
"""

        with connection.cursor() as cursor:
            execute_values(
                cursor,
                insert_query,
                records,
                page_size=500,
            )

        connection.commit()

        # Verify the records in PostgreSQL
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(*),
                    COUNT(DISTINCT city_name)
                FROM weather_forecasts;
                """
            )

            total_rows, total_cities = cursor.fetchone()

        print("\nWEATHER DATA LOAD SUCCESSFUL")
        print(f"CSV records processed: {len(records)}")
        print(f"Total records in database: {total_rows}")
        print(f"Cities in database: {total_cities}")

    except Exception:
        if connection is not None:
            connection.rollback()
        raise

    finally:
        if connection is not None:
            connection.close()
            print("Database connection closed.")


if __name__ == "__main__":
    load_weather_data()