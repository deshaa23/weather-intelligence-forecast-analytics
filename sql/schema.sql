-- PostgreSQL starter schema. Run after creating the weather_intelligence database.
CREATE TABLE IF NOT EXISTS dim_city (
    city_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    city_name VARCHAR(100) NOT NULL,
    state_name VARCHAR(100),
    country_code CHAR(2) NOT NULL DEFAULT 'IN',
    latitude NUMERIC(9, 6) NOT NULL,
    longitude NUMERIC(9, 6) NOT NULL,
    timezone VARCHAR(64) NOT NULL DEFAULT 'Asia/Kolkata',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    CONSTRAINT uq_dim_city_location UNIQUE (city_name, state_name, country_code)
);

CREATE TABLE IF NOT EXISTS fact_weather_forecast (
    forecast_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    city_id BIGINT NOT NULL REFERENCES dim_city(city_id),
    provider VARCHAR(40) NOT NULL,
    forecast_issued_at TIMESTAMPTZ NOT NULL,
    forecast_for TIMESTAMP NOT NULL,
    temperature_forecast_c NUMERIC(5, 2),
    humidity_forecast_pct NUMERIC(5, 2),
    precipitation_probability_pct NUMERIC(5, 2),
    precipitation_forecast_mm NUMERIC(8, 2),
    wind_speed_forecast_kmh NUMERIC(7, 2),
    weather_code INTEGER,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_forecast_humidity CHECK (
        humidity_forecast_pct IS NULL OR humidity_forecast_pct BETWEEN 0 AND 100
    ),
    CONSTRAINT ck_forecast_rain_probability CHECK (
        precipitation_probability_pct IS NULL
        OR precipitation_probability_pct BETWEEN 0 AND 100
    ),
    CONSTRAINT ck_forecast_precipitation CHECK (
        precipitation_forecast_mm IS NULL OR precipitation_forecast_mm >= 0
    ),
    CONSTRAINT ck_forecast_wind CHECK (
        wind_speed_forecast_kmh IS NULL OR wind_speed_forecast_kmh >= 0
    )
);

CREATE INDEX IF NOT EXISTS ix_forecast_city_target
    ON fact_weather_forecast (city_id, forecast_for);

CREATE INDEX IF NOT EXISTS ix_forecast_provider_issued
    ON fact_weather_forecast (provider, forecast_issued_at);

CREATE TABLE IF NOT EXISTS api_run_log (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    provider VARCHAR(40) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL,
    response_status INTEGER,
    records_received INTEGER NOT NULL DEFAULT 0,
    records_inserted INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    CONSTRAINT ck_api_run_status CHECK (status IN ('running', 'success', 'failed'))
);
