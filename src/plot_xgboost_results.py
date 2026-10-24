from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "reports" / "xgboost_test_predictions.csv"
REPORT_DIR = ROOT / "reports"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Test predictions not found: {INPUT_PATH}. "
            "Run python -m src.train_xgboost first."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, parse_dates=["target_timestamp"])

    required = {
        "target_timestamp",
        "actual_temperature_c",
        "xgboost_prediction_c",
        "city_name",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(
        subset=["actual_temperature_c", "xgboost_prediction_c"]
    )

    # Chart 1: Actual vs predicted temperature
    actual = df["actual_temperature_c"]
    predicted = df["xgboost_prediction_c"]

    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())

    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.25, s=10)
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Actual ERA5 temperature (°C)")
    plt.ylabel("XGBoost predicted temperature (°C)")
    plt.title("Actual vs Predicted Next-Hour Temperature")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "xgboost_actual_vs_predicted.png",
        dpi=150,
    )
    plt.close()

    # Chart 2: Prediction error distribution
    df["prediction_error_c"] = (
        df["xgboost_prediction_c"] - df["actual_temperature_c"]
    )

    plt.figure(figsize=(8, 6))
    plt.hist(df["prediction_error_c"], bins=50)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Prediction error (°C)")
    plt.ylabel("Number of test records")
    plt.title("XGBoost Prediction Error Distribution")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "xgboost_error_distribution.png",
        dpi=150,
    )
    plt.close()

    print("Evaluation charts saved:")
    print(REPORT_DIR / "xgboost_actual_vs_predicted.png")
    print(REPORT_DIR / "xgboost_error_distribution.png")
    print(f"Test records plotted: {len(df):,}")


if __name__ == "__main__":
    main()