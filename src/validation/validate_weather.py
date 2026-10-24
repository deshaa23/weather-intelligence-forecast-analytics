"""Validate the quality of the combined Indian cities weather dataset."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "india_weather_forecast.csv"

EXPECTED_CITIES = {
    "Pune",
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Chennai",
    "Hyderabad",
    "Kolkata",
}

REQUIRED_COLUMNS = [
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


def main() -> None:
    print("=" * 60)
    print("WEATHER DATA QUALITY VALIDATION")
    print("=" * 60)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    # 1. Basic dataset checks
    print(f"\nDataset: {DATA_FILE.name}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))

    if missing_columns:
        print(f"\nFAIL: Missing required columns: {missing_columns}")
        return

    print("PASS: All required columns are present.")

    # 2. Missing values
    print("\n--- Missing Values ---")
    missing_values = df[REQUIRED_COLUMNS].isna().sum()
    print(missing_values.to_string())

    missing_total = int(missing_values.sum())
    print(f"Total missing values in required columns: {missing_total}")

    # 3. Duplicate records
    print("\n--- Duplicate Checks ---")
    duplicate_rows = int(df.duplicated().sum())
    duplicate_city_hours = int(
        df.duplicated(subset=["city_name", "forecast_for"]).sum()
    )

    print(f"Exact duplicate rows: {duplicate_rows}")
    print(f"Duplicate city and forecast-hour pairs: {duplicate_city_hours}")

    # 4. Numeric data types
    numeric_columns = [
        "temperature_forecast_c",
        "humidity_forecast_pct",
        "precipitation_probability_pct",
        "precipitation_forecast_mm",
        "wind_speed_forecast_kmh",
        "weather_code",
    ]

    print("\n--- Data Type Checks ---")
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in numeric_columns:
        print(f"{column}: {df[column].dtype}")

    # 5. Valid measurement ranges
    print("\n--- Measurement Range Checks ---")

    range_checks = {
        "humidity_forecast_pct": (
            df["humidity_forecast_pct"].between(0, 100)
        ),
        "precipitation_probability_pct": (
            df["precipitation_probability_pct"].between(0, 100)
        ),
        "precipitation_forecast_mm": (
            df["precipitation_forecast_mm"].ge(0)
        ),
        "wind_speed_forecast_kmh": (
            df["wind_speed_forecast_kmh"].ge(0)
        ),
        "temperature_forecast_c": (
            df["temperature_forecast_c"].between(-90, 60)
        ),
    }

    invalid_range_counts = {}

    for column, valid_mask in range_checks.items():
        invalid_count = int((~valid_mask & df[column].notna()).sum())
        invalid_range_counts[column] = invalid_count
        status = "PASS" if invalid_count == 0 else "CHECK"
        print(f"{status}: {column} — {invalid_count} invalid values")

    # 6. City coverage
    print("\n--- City Coverage ---")
    city_counts = df["city_name"].value_counts().sort_index()
    print(city_counts.to_string())

    actual_cities = set(df["city_name"].dropna().unique())
    missing_cities = sorted(EXPECTED_CITIES - actual_cities)
    unexpected_cities = sorted(actual_cities - EXPECTED_CITIES)

    print(f"Missing expected cities: {missing_cities or 'None'}")
    print(f"Unexpected cities: {unexpected_cities or 'None'}")

    # 7. Forecast timestamps
    print("\n--- Timestamp Checks ---")

    issued_at = pd.to_datetime(df["forecast_issued_at"], errors="coerce", utc=True)
    forecast_for = pd.to_datetime(df["forecast_for"], errors="coerce")

    print(f"Invalid forecast-issued timestamps: {issued_at.isna().sum()}")
    print(f"Invalid forecast timestamps: {forecast_for.isna().sum()}")

    # 8. Overall summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    checks_passed = (
        len(df) > 0
        and missing_total == 0
        and duplicate_rows == 0
        and duplicate_city_hours == 0
        and all(count == 0 for count in invalid_range_counts.values())
        and not missing_cities
        and not unexpected_cities
        and issued_at.notna().all()
        and forecast_for.notna().all()
    )

    if checks_passed:
        print("RESULT: PASSED")
        print("No issues were found by the configured validation checks.")
    else:
        print("RESULT: REVIEW REQUIRED")
        print("Review the checks above before using this dataset downstream.")

    print("\nNote: A successful validation checks the configured rules;")
    print("it does not independently prove that forecasts are accurate.")


if __name__ == "__main__":
    main()