@echo off
cd /d "D:\weather analysis\weather-intelligence-starter"

if not exist logs mkdir logs

echo. >> logs\daily_weather.log
echo ===== STARTED %date% %time% ===== >> logs\daily_weather.log

python -m src.pipeline >> logs\daily_weather.log 2>&1
if errorlevel 1 goto pipeline_failed

python -m src.validation.validate_weather >> logs\daily_weather.log 2>&1
if errorlevel 1 goto validation_failed

python -m src.load_weather_to_db >> logs\daily_weather.log 2>&1
if errorlevel 1 goto load_failed

echo SUCCESS: Completed %date% %time% >> logs\daily_weather.log
exit /b 0

:pipeline_failed
echo ERROR: Weather pipeline failed. >> logs\daily_weather.log
exit /b 1

:validation_failed
echo ERROR: Data validation failed. Database load skipped. >> logs\daily_weather.log
exit /b 1

:load_failed
echo ERROR: PostgreSQL data loading failed. >> logs\daily_weather.log
exit /b 1