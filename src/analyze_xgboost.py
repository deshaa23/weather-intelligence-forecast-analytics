import json
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "xgboost_next_hour_temperature.json"
FEATURE_PATH = ROOT / "models" / "feature_columns.json"
REPORT_DIR = ROOT / "reports"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature list not found: {FEATURE_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(FEATURE_PATH, encoding="utf-8") as file:
        features = json.load(file)

    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    output_path = REPORT_DIR / "xgboost_feature_importance.csv"
    importance.to_csv(output_path, index=False)

    print("\nTop 15 features by model importance:")
    print(importance.head(15).to_string(index=False))

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()