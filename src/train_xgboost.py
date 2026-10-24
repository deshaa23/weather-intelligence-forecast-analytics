from pathlib import Path
import json

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed"
    / "historical_weather_reanalysis.csv"
)

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "xgboost_next_hour_temperature.json"
FEATURE_PATH = MODEL_DIR / "feature_columns.json"
METRICS_PATH = REPORT_DIR / "xgboost_metrics.csv"


# --------------------------------------------------
# 2. Load and check data
# --------------------------------------------------

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    df["timestamp_local"] = pd.to_datetime(
        df["timestamp_local"],
        errors="raise",
    )

    df = df.sort_values(
        ["city_name", "timestamp_local"]
    ).reset_index(drop=True)

    required = [
        "city_name",
        "timestamp_local",
        "temperature_2m_c",
        "relative_humidity_2m_pct",
        "precipitation_mm",
        "wind_speed_10m_kmh",
        "weather_code",
    ]

    missing_columns = set(required) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.duplicated(["city_name", "timestamp_local"]).any():
        raise ValueError("Duplicate city and timestamp records found.")

    print(f"Loaded rows: {len(df):,}")
    print(f"Cities: {df['city_name'].nunique()}")


    # --------------------------------------------------
    # 3. Build the next-hour prediction target
    # --------------------------------------------------

    grouped = df.groupby("city_name", sort=False)

    df["target_timestamp"] = grouped["timestamp_local"].shift(-1)

    df["target_temperature_next_hour_c"] = (
        grouped["temperature_2m_c"].shift(-1)
    )

    # Exclude rows where the following record is not
    # exactly one hour later.
    consecutive_hour = (
        df["target_timestamp"]
        == df["timestamp_local"] + pd.Timedelta(hours=1)
    )

    df = df.loc[consecutive_hour].copy()


    # --------------------------------------------------
    # 4. Feature engineering
    # --------------------------------------------------

    grouped = df.groupby("city_name", sort=False)

    # Past temperature values
    df["temperature_lag_1h_c"] = (
        grouped["temperature_2m_c"].shift(1)
    )

    df["temperature_lag_24h_c"] = (
        grouped["temperature_2m_c"].shift(24)
    )

    # Rolling averages use current and previous observations.
    df["temperature_rolling_3h_c"] = (
        df.groupby("city_name")["temperature_2m_c"]
        .transform(lambda s: s.rolling(3, min_periods=3).mean())
    )

    df["temperature_rolling_24h_c"] = (
        df.groupby("city_name")["temperature_2m_c"]
        .transform(lambda s: s.rolling(24, min_periods=24).mean())
    )

    # Time features from the input hour
    hour = df["timestamp_local"].dt.hour
    day_of_year = df["timestamp_local"].dt.dayofyear
    day_of_week = df["timestamp_local"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )
    df["day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    df["day_of_week"] = day_of_week

    numeric_features = [
        "temperature_2m_c",
        "relative_humidity_2m_pct",
        "precipitation_mm",
        "wind_speed_10m_kmh",
        "temperature_lag_1h_c",
        "temperature_lag_24h_c",
        "temperature_rolling_3h_c",
        "temperature_rolling_24h_c",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "day_of_week",
    ]

    # Treat city and weather code as categories, not numbers.
    city_features = pd.get_dummies(
        df["city_name"],
        prefix="city",
        dtype=int,
    )

    weather_features = pd.get_dummies(
        df["weather_code"].astype(str),
        prefix="weather_code",
        dtype=int,
    )

    X = pd.concat(
        [
            df[numeric_features].reset_index(drop=True),
            city_features.reset_index(drop=True),
            weather_features.reset_index(drop=True),
        ],
        axis=1,
    )

    y = df["target_temperature_next_hour_c"].reset_index(drop=True)
    timestamps = df["timestamp_local"].reset_index(drop=True)
    target_timestamps = df["target_timestamp"].reset_index(drop=True)
    cities = df["city_name"].reset_index(drop=True)

    # Drop warm-up rows that lack sufficient lag history.
    valid = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    timestamps = timestamps.loc[valid].reset_index(drop=True)
    target_timestamps = target_timestamps.loc[valid].reset_index(drop=True)
    cities = cities.loc[valid].reset_index(drop=True)

    print(f"Usable modeling rows: {len(X):,}")


    # --------------------------------------------------
    # 5. Chronological train/test split
    # --------------------------------------------------

    unique_times = np.sort(timestamps.unique())

    split_index = int(len(unique_times) * 0.80)
    cutoff = pd.Timestamp(unique_times[split_index])

    # Training targets must also be before the cutoff.
    train_mask = (
        (timestamps < cutoff)
        & (target_timestamps < cutoff)
    )

    # Test on later input timestamps.
    test_mask = timestamps >= cutoff

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    if X_train.empty or X_test.empty:
        raise ValueError("The chronological split produced an empty set.")

    print(f"Training cutoff: {cutoff}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")


    # --------------------------------------------------
    # 6. Baseline and XGBoost
    # --------------------------------------------------

    # Baseline: next-hour temperature equals current temperature.
    baseline_predictions = X_test["temperature_2m_c"]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    print("\nTraining XGBoost...")
    model.fit(X_train, y_train)

    xgb_predictions = model.predict(X_test)
    
        # Save test predictions and evaluate each city
    test_predictions = pd.DataFrame({
        "city_name": cities.loc[test_mask].reset_index(drop=True),
        "input_timestamp": timestamps.loc[test_mask].reset_index(drop=True),
        "target_timestamp": target_timestamps.loc[test_mask].reset_index(drop=True),
        "actual_temperature_c": y_test.to_numpy(),
        "baseline_prediction_c": baseline_predictions.to_numpy(),
        "xgboost_prediction_c": xgb_predictions,
    })

    city_results = []

    for city, group in test_predictions.groupby("city_name"):
        actual = group["actual_temperature_c"]

        baseline_pred = group["baseline_prediction_c"]
        xgb_pred = group["xgboost_prediction_c"]

        baseline_mae = mean_absolute_error(actual, baseline_pred)
        xgb_mae = mean_absolute_error(actual, xgb_pred)

        baseline_rmse = np.sqrt(
            mean_squared_error(actual, baseline_pred)
        )
        xgb_rmse = np.sqrt(
            mean_squared_error(actual, xgb_pred)
        )

        improvement = (
            (baseline_mae - xgb_mae) / baseline_mae * 100
            if baseline_mae > 0
            else np.nan
        )

        city_results.append({
            "city_name": city,
            "test_records": len(group),
            "baseline_MAE_C": baseline_mae,
            "xgboost_MAE_C": xgb_mae,
            "baseline_RMSE_C": baseline_rmse,
            "xgboost_RMSE_C": xgb_rmse,
            "MAE_improvement_pct": improvement,
        })

    city_metrics = pd.DataFrame(city_results)


    # --------------------------------------------------
    # 7. Evaluate
    # --------------------------------------------------

    def calculate_metrics(actual, predicted):
        return {
            "MAE_C": mean_absolute_error(actual, predicted),
            "RMSE_C": np.sqrt(mean_squared_error(actual, predicted)),
        }

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_predictions,
    )

    xgb_metrics = calculate_metrics(
        y_test,
        xgb_predictions,
    )

    results = pd.DataFrame([
        {
            "model": "Persistence baseline",
            **baseline_metrics,
        },
        {
            "model": "XGBoost",
            **xgb_metrics,
        },
    ])

    # Percentage reduction in MAE vs baseline, if defined.
    baseline_mae = baseline_metrics["MAE_C"]
    improvement = (
        (baseline_mae - xgb_metrics["MAE_C"]) / baseline_mae * 100
        if baseline_mae > 0
        else np.nan
    )

    print("\nEvaluation results (degrees Celsius)")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nXGBoost MAE improvement vs baseline: {improvement:.2f}%")


    # --------------------------------------------------
    # 8. Save model and report
    # --------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(MODEL_PATH))

    with open(FEATURE_PATH, "w", encoding="utf-8") as file:
        json.dump(list(X.columns), file, indent=2)

    results.to_csv(METRICS_PATH, index=False)
    
    city_metrics.to_csv(
        REPORT_DIR / "xgboost_metrics_by_city.csv",
        index=False,
    )

    test_predictions.to_csv(
        REPORT_DIR / "xgboost_test_predictions.csv",
        index=False,
    )

    print("\nEvaluation by city:")
    print(
        city_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved model:", MODEL_PATH)
    print("Saved feature list:", FEATURE_PATH)
    print("Saved evaluation:", METRICS_PATH)


if __name__ == "__main__":
    main()