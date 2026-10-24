## Machine Learning: XGBoost Temperature Prediction

### Objective
Predict the next hour's temperature using historical hourly weather
data from seven Indian cities.

### Dataset
- Source: Open-Meteo Historical Weather API
- Model/data type: ERA5 reanalysis
- Period: January 2024 – December 2025
- Total records collected: 122,808
- Cities: Pune, Mumbai, Delhi, Bengaluru, Chennai, Hyderabad, Kolkata
- Target: Next-hour temperature (°C)

### Methodology
- Engineered temperature lag and rolling-average features.
- Added time-based features to capture daily and seasonal patterns.
- Used a chronological train/test split to evaluate later timestamps.
- Compared XGBoost Regression against a persistence baseline.
- Evaluated performance using MAE and RMSE.

### Evaluation Results

| Model | MAE (°C) | RMSE (°C) |
|---|---:|---:|
| Persistence Baseline | 0.6543 | 1.0033 |
| XGBoost Regressor | 0.2908 | 0.4061 |

XGBoost reduced MAE by 55.56% relative to the baseline on the
held-out test set.

### Evaluation Visualizations
- `reports/xgboost_actual_vs_predicted.png`
- `reports/xgboost_error_distribution.png`
- `reports/xgboost_metrics.csv`
- `reports/xgboost_metrics_by_city.csv`
- `reports/xgboost_feature_importance.csv`

### Limitations
The evaluation targets are ERA5 reanalysis values, not independent
weather-station observations. The results demonstrate performance
on this dataset and should not be interpreted as verified
real-world weather forecast accuracy.