# Weather Intelligence Dashboard (Streamlit)

This is a code-built, interactive dashboard inspired by the supplied reference image. It uses the user's existing Weather Intelligence project data; the KPI values and charts are read from PostgreSQL and generated model evaluation CSV files rather than hardcoded mock numbers.

## Place the files

Copy `dashboard_app.py` and `requirements-dashboard.txt` into the root of the existing project:

`D:\weather analysis\weather-intelligence-starter`

The root should already contain `.env`, `data`, `models`, `reports`, and `src`.

## Install dependencies

Open PowerShell:

```powershell
cd "D:\weather analysis\weather-intelligence-starter"
python -m pip install -r requirements-dashboard.txt
```

## Run

```powershell
streamlit run dashboard_app.py
```

Streamlit opens the dashboard in a browser. Keep PostgreSQL running and ensure the existing root `.env` contains:

```dotenv
DB_HOST=localhost
DB_PORT=5432
DB_NAME=weather_intelligence
DB_USER=postgres
DB_PASSWORD=your_actual_password
```

Do not upload `.env` to GitHub. The app reads it locally and does not display the password.

## Data sources

- `public.weather_forecasts`: current forecast and retained forecast snapshots.
- `reports/xgboost_test_predictions.csv`: timestamp-level actual ERA5 target vs. XGBoost prediction.
- `reports/xgboost_metrics.csv`: overall persistence baseline and XGBoost MAE/RMSE.
- `reports/xgboost_metrics_by_city.csv`: city-level MAE/RMSE comparison.
- `reports/xgboost_feature_importance.csv`: feature importance report.

If PostgreSQL is unavailable, the app attempts to use `data/processed/india_weather_forecast.csv` for the current forecast overview; historical snapshot comparisons require PostgreSQL.

## Notes

- The app expects the project's existing table and model report columns.
- Streamlit caches PostgreSQL data for up to five minutes. Use the app's **Rerun** command to force a fresh app run; data cache expiry may still apply.
- Model metrics were evaluated against held-out ERA5 reanalysis targets, not independent weather-station observations. Avoid describing these as verified real-world forecast accuracy.
